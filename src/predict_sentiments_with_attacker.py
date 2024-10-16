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

import numpy as np
import pandas as pd
import argparse
from openai import OpenAI
from data_loader import load_dataset, subsample_random_balanced_dataset



# Add the arguments and parse them.
parser = argparse.ArgumentParser(description="Predicts the financial sentiment class for all the samples of a given dataset using the attacker model (GPT-4o).")
parser.add_argument('--dataset_name', type=str, help='A dataset string to choose from the set {"FPB", "TFNS", "SEntFiN"}', required=True)
parser.add_argument('--openai_api_key', type=str, help='The string containing the OpenAI key.', required=True)
args = parser.parse_args()


# Input global parameters.
dataset_name = args.dataset_name
openai_api_key = args.openai_api_key


# Other global parameters, some of which can be used for debugging or continuing a failed job due to OpenAI API issues.
version_text = 'v8'
sample_starting_index = 0 # 0 or another index to continue a broken processing work!
random_balanced_subsampling = False # if set to True, we sample a smaller dataset where each sentiment class has equal number of samples
no_samples_to_process = None # a value or None to process the entire dataset!
if dataset_name == "FPB":
    in_dataset_path = "../datasets/financial-phrasebank/Sentences_AllAgree.txt"
    out_dataset_path = "../results/financial-phrasebank/gpt4o_sentiment_"+version_text+".csv"
elif dataset_name == "TFNS":
    in_dataset_path = "../datasets/twitter-financial-news-sentiment/sent_valid.csv"
    out_dataset_path = "../results/twitter-financial-news-sentiment/gpt4o_sentiment_"+version_text+".csv"
elif dataset_name == "SEntFiN":
    random_balanced_subsampling = True
    no_samples_to_process = 2100
    in_dataset_path = "../datasets/SEntFiN/SEntFiN.csv"
    out_dataset_path = "../results/SEntFiN/gpt4o_sentiment_"+version_text+".csv"
else:
    raise ValueError('Dataset name ' + dataset_name + ' is not recognized!')

# Attacking Model (GPT-4o)-related parameters.
client = OpenAI(api_key=openai_api_key)
prompt_model = "gpt-4o"
random_seed = None
no_of_predictions_per_sample = 3

instruction = "Does the following text have a POSITIVE, NEGATIVE or NEUTRAL financial sentiment based only on the facts stated in the text? Output a single-word response representing the sentiment class:\n\n"





# Load the dataset.
dataset = load_dataset(dataset_name, in_dataset_path)

if no_samples_to_process is None:
    no_samples_to_process = len(dataset)
    
# Randomly sample a subset with balanced number of sentiment classes.
if random_balanced_subsampling:
    dataset = subsample_random_balanced_dataset(dataset, no_samples_to_process // (dataset['label'].nunique()))


# Create the necessary additional columns.
for ti in range(no_of_predictions_per_sample):
    dataset['gpt_pred_'+str(ti)] = -1
    dataset['gpt_response_'+str(ti)] = ""
dataset['gpt_pred'] = -1 # combined (winner-takes all) response of gpt

if sample_starting_index == 0:
    # Create a new empty csv file.
    out_dataset_file = open(out_dataset_path,'w')
    out_dataset_file.close()


for si in range(sample_starting_index, min(len(dataset), sample_starting_index+no_samples_to_process)):

    print("Processing sample ", si)

    fin_text = dataset.iloc[si].text
    
    all_predictions = np.zeros((3), np.int32) # sentiment histogram for predictions
    for ti in range(no_of_predictions_per_sample):
        dataset['gpt_pred_'+str(ti)] = -1
        dataset['gpt_response_'+str(ti)] = ""

        # Predict the sentiment class with chatgpt using a low temperature and top_p. 
        # It has been reported that this improves determinism and logical functions, which is imprortant for certain tasks such as code generation.
        sentiment_str = client.chat.completions.create(model=prompt_model, seed = random_seed, temperature=0.1, top_p=0.1, max_tokens=3, messages=[{"role": "user", "content": instruction +  fin_text}])

        sentiment_str = sentiment_str.choices[0].message.content.strip()
        if "NEUTRAL" in sentiment_str:
            sentiment = 2
        elif "POSITIVE" in sentiment_str:
            sentiment = 0
        elif "NEGATIVE" in sentiment_str:
            sentiment = 1
        else:
            sentiment = -1

        if sentiment > 0:
            all_predictions[sentiment] += 1

        dataset.loc[si, 'gpt_response_'+str(ti)] = sentiment_str
        dataset.loc[si, 'gpt_pred_'+str(ti)] = sentiment

    dataset.loc[si, 'gpt_pred'] = np.argmax(all_predictions) # combined (winner-takes all) response of gpt

    # Save these similar prompts to the output dataset file. We do it row by row not to loose the entire
    # dataset if or when the application crashes for some reason.
    dataset.loc[[si]].to_csv(out_dataset_path, index=False, header=(si==0), mode='a')
