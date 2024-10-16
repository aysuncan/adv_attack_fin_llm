"""
   Copyright 2024 Aysun Can Türetken

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import struct
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from openai import OpenAI
from data_loader import load_dataset, subsample_random_balanced_dataset
pd.options.mode.chained_assignment = None  # default='warn'


# Add the arguments and parse them.
parser = argparse.ArgumentParser(description="Creates paraphrased samples and their GPT embeddings on a given dataset.")
parser.add_argument('--dataset_name', type=str, help='A dataset string to choose from the set {"FPB", "TFNS", "SEntFiN"}', required=True)
parser.add_argument('--openai_api_key', type=str, help='The string containing the OpenAI key.', required=True)
parser.add_argument('--no_paraphrases', type=int, help='The number of paraphrases to be generated for each sample in the given dataset.', default=25, required=False)
args = parser.parse_args()


# Input global parameters.
dataset_name = args.dataset_name
openai_api_key = args.openai_api_key
no_paraphrases = args.no_paraphrases


# Other global parameters, some of which can be used for debugging or continuing a failed job due to OpenAI API issues.
version_text = 'v8'
sample_starting_index = 0 # 0 or another index to continue a broken processing work!
random_balanced_subsampling = False # if set to True, we sample a smaller dataset where each sentiment class has equal number of samples
no_samples_to_process = None # a value or None to process the entire dataset!
if dataset_name == "FPB":
    in_dataset_path = "../datasets/financial-phrasebank/Sentences_AllAgree.txt"
    out_dataset_path = "../results/financial-phrasebank/Prompts_AllAgree_"+version_text+".csv"
    out_embedding_path = "../results/financial-phrasebank/Embeddings_AllAgree_"+version_text+".bin"
elif dataset_name == "TFNS":
    in_dataset_path = "../datasets/twitter-financial-news-sentiment/sent_valid.csv"
    out_dataset_path = "../results/twitter-financial-news-sentiment/prompts_valid_"+version_text+".csv"
    out_embedding_path = "../results/twitter-financial-news-sentiment/embeddings_valid_"+version_text+".bin"
elif dataset_name == "SEntFiN":
    random_balanced_subsampling = True
    no_samples_to_process = 2100
    in_dataset_path = "../datasets/SEntFiN/SEntFiN.csv"
    out_dataset_path = "../results/SEntFiN/paraphrases_SEntFiN_"+version_text+".csv"
    out_embedding_path = "../results/SEntFiN/embeddings_SEntFiN_"+version_text+".bin"
else:
     raise ValueError('Dataset name ' + dataset_name + ' is not recognized!')

# Attacking Model (GPT4o)-related parameters.
client = OpenAI(api_key=openai_api_key)
prompt_model = "gpt-4o"
embedding_model = "text-embedding-3-large"
random_seed = None
max_no_trials_for_sim_prompt_gen = 3




# Create the instruction.
instruction = "Rephrase the following financial text in " + str(no_paraphrases) + " different ways and present your response in a numbered list starting at 1. Be concise and make sure that each paraphrase has exactly the same factual information and financial sentiment:\n\n"


# Load the dataset.
dataset = load_dataset(dataset_name, in_dataset_path)

if no_samples_to_process is None:
    no_samples_to_process = len(dataset)

# Randomly sample a subset with balanced number of sentiment classes.
if random_balanced_subsampling:
    dataset = subsample_random_balanced_dataset(dataset, no_samples_to_process // (dataset['label'].nunique()))




# Create the output dataset (dataframe) which will contain the similar prompts to be used for attacking.
out_dataset = dataset.copy()

for i in range(no_paraphrases):
    out_dataset['similar_text' + str(i)] = ''

if sample_starting_index == 0:
    # Create a new empty csv and binary embedding files.
    out_embedding_file = open(out_embedding_path,'wb')
    out_embedding_file.close()
    out_dataset_file = open(out_dataset_path,'w')
    out_dataset_file.close()

# Open the empty file in append mode to write the embeddings of the attacking model.
out_embedding_file = open(out_embedding_path,'ab')
    

for si in range(sample_starting_index, min(len(dataset), sample_starting_index+no_samples_to_process)):

    print("Processing sample ", si)

    fin_text = dataset.iloc[si].text
    fin_text_label = dataset.iloc[si].label

    # Create similar prompts. If the number of prompts generated is not equal to no_paraphrases, continue trying/generating until it is.
    similar_prompt_list = []
    no_of_trials = 0
    while (len(similar_prompt_list) < no_paraphrases) and (no_of_trials < max_no_trials_for_sim_prompt_gen):
        similar_prompt_list = []
        # Create similar prompts to be used for attack using the attacking model. The output is a single long string.
        similar_prompts = client.chat.completions.create(model=prompt_model, seed = random_seed, messages=[{"role": "user", "content": instruction +  fin_text}])
        similar_prompts_str = similar_prompts.choices[0].message.content

        # Extract each similar prompt individually from the bulk string output of the attacking model.
        start_search_str = '1. '
        start_ind = similar_prompts_str.find(start_search_str)
        if start_ind == -1:
            start_search_str = '1) '
            start_ind = similar_prompts_str.find(start_search_str)
        if start_ind != -1:
            for i in range(2, no_paraphrases+1):
                end_search_str = "\n" + str(i) + '. '
                end_ind = similar_prompts_str.find(end_search_str)
                if end_ind == -1:
                    end_search_str = "\n" + str(i) + ') '
                    end_ind = similar_prompts_str.find(end_search_str)
                if end_ind == -1:
                    continue
                similar_prompt = similar_prompts_str[start_ind + len(start_search_str):end_ind]
                similar_prompt_list.append(similar_prompt)
                start_search_str = end_search_str
                start_ind = end_ind
            # Add the last similar prompt to our list.
            similar_prompt_list.append(similar_prompts_str[start_ind + len(start_search_str):])

        no_of_trials = no_of_trials + 1
        if len(similar_prompt_list) < no_paraphrases:
            print("Sample", str(si), "resulted in only", len(similar_prompt_list), "similar prompts. Trying again to generate", str(no_paraphrases), "prompts.")
    
    # Add the generated similar prompts to our output dataset.
    for i in range(len(similar_prompt_list)):
        out_dataset.loc[si, 'similar_text' + str(i)] = similar_prompt_list[i]

    # Extract the embeddings of the attacking model on all the prompts.
    attacking_embedding_list = [np.array(client.embeddings.create(input=fin_text, model=embedding_model).data[0].embedding)]
    for i in tqdm(range(len(similar_prompt_list))):
        similar_prompt = similar_prompt_list[i]
        response = client.embeddings.create(input=similar_prompt, model=embedding_model)
        attacking_embedding_list.append(np.array(response.data[0].embedding))

    # Save these similar prompts to the output dataset file. We do it row by row not to loose the entire
    # dataset if or when the application crashes for some reason.
    out_dataset.loc[[si]].to_csv(out_dataset_path, index=False, header=(si==0), mode='a')

    if si == 0:    
        # Write the total number of samples and the embedding length. 
        # This is done only once at the beggining of the binary file.
        out_embedding_file.write(struct.pack('Q', no_samples_to_process))
        out_embedding_file.write(struct.pack('Q', len(attacking_embedding_list[0])))
        out_embedding_file.flush()

    # Now, write the number of embeddings (i.e., number of similar text for each original sample) 
    # for this sample as well as the associated embedding values.
    out_embedding_file.write(struct.pack('Q', np.uint64(len(attacking_embedding_list))))
    for ai in range(len(attacking_embedding_list)): # write the embeddings for this sample. The first embedding is that of the original text.
        attacking_embedding = attacking_embedding_list[ai]
        for ei in range(len(attacking_embedding)):
            out_embedding_file.write(struct.pack('d', attacking_embedding[ei]))
    out_embedding_file.flush()

out_embedding_file.close()
