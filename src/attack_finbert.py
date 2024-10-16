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
import sys
import struct
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from openai import OpenAI
from finbert_embedding import FinbertEmbedding



# Add the arguments and parse them.
parser = argparse.ArgumentParser(description="Perform attack on the FinBERT model over the pre-processed samples of a given dataset.")
parser.add_argument('--dataset_name', type=str, help='A dataset string to choose from the set {"FPB", "TFNS", "SEntFiN"}', required=True)
parser.add_argument('--openai_api_key', type=str, help='The string containing the OpenAI key.', required=True)
args = parser.parse_args()


# Input global parameters.
dataset_name = args.dataset_name
openai_api_key = args.openai_api_key


# Other global parameters.
version_text = 'v8'
verbose = True
max_no_filtering_trials = 3
prompt_model = "gpt-4o"
client = OpenAI(api_key=openai_api_key)

filter_instruction = "Does the following text have a POSITIVE, NEGATIVE or NEUTRAL financial sentiment based only on the facts stated in the text? Output a single-word response representing the sentiment class:\n\n"


attacked_model_path = "../models/finbert" # this is the financial sentiment classification model which outputs probabilities for (positive, negative and neutral) sentiments.
if dataset_name == "FPB":
    sim_prompt_dataset_path = "../results/financial-phrasebank/Prompts_AllAgree_"+version_text+".csv"
    gpt_prediction_dataset_path = "../results/financial-phrasebank/gpt4o_sentiment_"+version_text+".csv"
    sim_prompt_embeddings_path = "../results/financial-phrasebank/Embeddings_AllAgree_"+version_text+".bin"
    output_results_path = "../results/financial-phrasebank/performance_results_"+version_text+"_FinBERT.csv"
elif dataset_name == "TFNS":
    sim_prompt_dataset_path = "../results/twitter-financial-news-sentiment/prompts_valid_"+version_text+".csv"
    gpt_prediction_dataset_path = "../results/twitter-financial-news-sentiment/gpt4o_sentiment_"+version_text+".csv"
    sim_prompt_embeddings_path = "../results/twitter-financial-news-sentiment/embeddings_valid_"+version_text+".bin"
    output_results_path = "../results/twitter-financial-news-sentiment/performance_results_"+version_text+"_FinBERT.csv"
elif dataset_name == "SEntFiN":
    sim_prompt_dataset_path = "../results/SEntFiN/paraphrases_SEntFiN_"+version_text+".csv"
    gpt_prediction_dataset_path = "../results/SEntFiN/gpt4o_sentiment_"+version_text+".csv"
    sim_prompt_embeddings_path = "../results/SEntFiN/embeddings_SEntFiN_"+version_text+".bin"
    output_results_path = "../results/SEntFiN/performance_results_"+version_text+"_FinBERT.csv"
else:
    raise ValueError('Dataset name ' + dataset_name + ' is not recognized!')




# Create and load the finbert model to perform both embedding vector extraction and financial sentiment analysis.
finbert = FinbertEmbedding(attacked_model_path)

# Load the dataset including both the original and similar (paraphrase) prompts to be used in attack.
dataset = pd.read_csv(sim_prompt_dataset_path)

# Load the dataset containing the predictions of the attacking model and add these precistions to the above dataset.
attacking_model_preds = pd.read_csv(gpt_prediction_dataset_path)
dataset['gpt_pred'] = attacking_model_preds['gpt_pred']


# Open the empty file in append mode to write the embeddings of the attacking model.
sim_prompt_embeddings_file = open(sim_prompt_embeddings_path,'rb')


# Read the total number of samples and the embedding length. 
# This info is present only at the beggining of the binary file.
no_of_samples_to_process = struct.unpack('Q', sim_prompt_embeddings_file.read(8))[0]
embedding_length = struct.unpack('Q', sim_prompt_embeddings_file.read(8))[0]
if no_of_samples_to_process != len(dataset):
    print("WARNING: The number of samples recorded in the embeddings file is",no_of_samples_to_process, "which is different than the number of samples", len(dataset), "present in the csv file. Processing only",min(no_of_samples_to_process,len(dataset)),"samples!")
    no_of_samples_to_process = min(no_of_samples_to_process, len(dataset))
print("DATASET CONTAINS ", no_of_samples_to_process, " ORIGINAL SAMPLES!")
if verbose:
    print("VERBOSE MODE ON! PRINTING RESULTS AS THEY ARE GENERATED ...")
print("\n")
    
# Go over all the samples in the dataset.
orig_accuracy = 0
attack_accuracy = 0
flipped_pred_rate = 0
global_result_list = []
for si in range(no_of_samples_to_process):

    # Extract the number of similar (paraphrase) prompts for this sample.
    # This number includes the original text, which is why we subtract one from it.
    no_of_simil_prompts = struct.unpack('Q', sim_prompt_embeddings_file.read(8))[0] - 1


    # Extract the original text and its embedding of the attacking model.
    orig_label = dataset['label'].iloc[si]
    orig_prompt = dataset['text'].iloc[si]
    orig_gpt_embedding = np.array(struct.unpack(f'{embedding_length}d', sim_prompt_embeddings_file.read(embedding_length * 8)), dtype=np.float64)
    orig_gpt_pred = dataset['gpt_pred'].iloc[si]

    # Extract the similar (paraphrase) texts and their embeddings of the attacking model.
    simil_prompts = [None] * no_of_simil_prompts
    simil_gpt_embeddings = [None] * no_of_simil_prompts
    for i in range(no_of_simil_prompts):
        simil_prompts[i] = dataset['similar_text' + str(i)].iloc[si]
        # Cut the text prompt from the newline if it exists.
        newline_index = simil_prompts[i].find('\n')
        if newline_index >= 0:
            simil_prompts[i] = simil_prompts[i][:newline_index]
        simil_gpt_embeddings[i] = np.array(struct.unpack(f'{embedding_length}d', sim_prompt_embeddings_file.read(embedding_length * 8)), dtype=np.float64)


    # Extract the embedding and sentiment class probabilities for the original text from the model to be attacked.
    orig_attacked_model_embedding, orig_attacked_model_prob = finbert.get_embedding(orig_prompt)
    orig_attacked_model_embedding = orig_attacked_model_embedding[0].numpy()
    orig_attacked_model_embedding /= np.linalg.norm(orig_attacked_model_embedding)
    orig_attacked_model_prob = list(orig_attacked_model_prob.numpy()[0])

    # Extract the embeddings and sentiment class probabilities for the similar (paraphrase) texts from the model to be attacked.
    simil_attacked_model_embeddings = [None] * no_of_simil_prompts
    simil_attacked_model_probs = [None] * no_of_simil_prompts
    for i in range(no_of_simil_prompts):
        embedding, prob = finbert.get_embedding(simil_prompts[i])
        embedding = embedding[0].numpy()
        embedding /= np.linalg.norm(embedding)
        prob = prob.numpy()
        simil_attacked_model_embeddings[i] = embedding
        simil_attacked_model_probs[i] = prob


    # Collect the results.
    sample_result_list = []
    for i in range(no_of_simil_prompts):
        gpt_simil = np.dot(orig_gpt_embedding, simil_gpt_embeddings[i]) + 1.0
        attacked_model_simil = np.dot(orig_attacked_model_embedding, simil_attacked_model_embeddings[i]) + 1.0
        sample_result_list.append({'simil_ratio': gpt_simil / (attacked_model_simil + sys.float_info.epsilon), 
                                   'gpt_simil': gpt_simil, 'attacked_model_simil': attacked_model_simil,  
                                   'attacked_model_sentiment_probs': list(simil_attacked_model_probs[i][0]), 
                                   'attack_text': simil_prompts[i]})

    # Sort the result list according to the embedding similarity ratio.
    sample_result_list.sort(key=lambda x: x['simil_ratio'], reverse=True)

    # Create the final result dictionary for this sample.
    result_cols = ['orig_text', 'orig_label', 'orig_gpt_pred', 'orig_pred', 'orig_sentiment_probs']
    sample_result_dict = {result_cols[0]: orig_prompt, result_cols[1]: orig_label, result_cols[2]: orig_gpt_pred,
                          result_cols[3]: np.argmax(orig_attacked_model_prob), result_cols[4]: orig_attacked_model_prob}

    # Populate attack text related fields of the result dictionary.
    bestValidAttackIndex = 0
    bestValidAttackFound = False
    for i in range(max_no_filtering_trials):
        top_k_attack_result = sample_result_list[i]

        result_cols.append('attack_text_'+str(i))
        sample_result_dict[result_cols[-1]] = top_k_attack_result['attack_text']
        result_cols.append('attack_text_gpt_pred_'+str(i))
        sample_result_dict[result_cols[-1]] = -1
        result_cols.append('attacked_model_pred_'+str(i))
        sample_result_dict[result_cols[-1]] = np.argmax(top_k_attack_result['attacked_model_sentiment_probs'])
        result_cols.append('attacked_model_sentiment_probs_'+str(i))
        sample_result_dict[result_cols[-1]] = top_k_attack_result['attacked_model_sentiment_probs']
        result_cols.append('gpt_simil_'+str(i))
        sample_result_dict[result_cols[-1]] = top_k_attack_result['gpt_simil']
        result_cols.append('attacked_model_simil_'+str(i))
        sample_result_dict[result_cols[-1]] = top_k_attack_result['attacked_model_simil']
        result_cols.append('simil_ratio_'+str(i))
        sample_result_dict[result_cols[-1]] = top_k_attack_result['simil_ratio']
    
        # Check if the currecnt attack text can pass our gpt filter to make sure that the sentiment did not change. 
        if not bestValidAttackFound:
            fin_text = sample_result_dict['attack_text_'+str(i)]
            sentiment_str = client.chat.completions.create(model=prompt_model, temperature=0.1, top_p=0.1, max_tokens=3, messages=[{"role": "user", "content": filter_instruction +  fin_text}])
            sentiment_str = sentiment_str.choices[0].message.content.strip()
            if "NEUTRAL" in sentiment_str:
                attack_text_gpt_pred = 2
            elif "POSITIVE" in sentiment_str:
                attack_text_gpt_pred = 0
            elif "NEGATIVE" in sentiment_str:
                attack_text_gpt_pred = 1
            else:
                attack_text_gpt_pred = -1

            # Does the current attack text pass our filter, which means it is a valid attack!
            sample_result_dict['attack_text_gpt_pred_'+str(i)] = attack_text_gpt_pred
            bestValidAttackFound = attack_text_gpt_pred == sample_result_dict['orig_gpt_pred']
            if bestValidAttackFound:
                bestValidAttackIndex = i
            

    # Create the necessary columns for the final selected attack text that passes the filter.
    # If we could not find such a valid attack text, we use the original text instead.
    if bestValidAttackFound:
        result_cols.append('attack_text')
        sample_result_dict[result_cols[-1]] = sample_result_dict['attack_text_'+str(bestValidAttackIndex)]
        result_cols.append('attack_text_gpt_pred')
        sample_result_dict[result_cols[-1]] = sample_result_dict['attack_text_gpt_pred_'+str(bestValidAttackIndex)]
        result_cols.append('attacked_model_pred')
        sample_result_dict[result_cols[-1]] = sample_result_dict['attacked_model_pred_'+str(bestValidAttackIndex)]
        result_cols.append('attacked_model_sentiment_probs')
        sample_result_dict[result_cols[-1]] = sample_result_dict['attacked_model_sentiment_probs_'+str(bestValidAttackIndex)]
        result_cols.append('gpt_simil')
        sample_result_dict[result_cols[-1]] = sample_result_dict['gpt_simil_'+str(bestValidAttackIndex)]
        result_cols.append('attacked_model_simil')
        sample_result_dict[result_cols[-1]] = sample_result_dict['attacked_model_simil_'+str(bestValidAttackIndex)]
        result_cols.append('simil_ratio')
        sample_result_dict[result_cols[-1]] = sample_result_dict['simil_ratio_'+str(bestValidAttackIndex)]
    else:
        result_cols.append('attack_text')
        sample_result_dict[result_cols[-1]] = sample_result_dict['orig_text']
        result_cols.append('attack_text_gpt_pred')
        sample_result_dict[result_cols[-1]] = sample_result_dict['orig_gpt_pred']
        result_cols.append('attacked_model_pred')
        sample_result_dict[result_cols[-1]] = sample_result_dict['orig_pred']
        result_cols.append('attacked_model_sentiment_probs')
        sample_result_dict[result_cols[-1]] = sample_result_dict['orig_sentiment_probs']
        result_cols.append('gpt_simil')
        sample_result_dict[result_cols[-1]] = 2.0
        result_cols.append('attacked_model_simil')
        sample_result_dict[result_cols[-1]] = 2.0
        result_cols.append('simil_ratio')
        sample_result_dict[result_cols[-1]] = 1.0

    # Convert dictionary values into a tuple in the specified key order
    global_result_list.append( tuple(sample_result_dict[key] for key in result_cols) )
    
    if verbose:
        # Compute running accuracies and print the results for this sample.
        orig_accuracy += int(sample_result_dict['orig_pred'] == orig_label)
        attack_accuracy += int(sample_result_dict['attacked_model_pred'] == orig_label)
        flipped_pred_rate += int(sample_result_dict['orig_pred'] != sample_result_dict['attacked_model_pred'])
        print("SAMPLE " + str(si))
        print("ORIG. TEXT:  ", sample_result_dict['orig_text'])
        print("ATTACK TEXT: ", sample_result_dict['attack_text'])
        print("GT & ORIG. PRED. & ATTACK PRED.: ",
                sample_result_dict['orig_label'],
                np.argmax(sample_result_dict['orig_sentiment_probs']),
                np.argmax(sample_result_dict['attacked_model_sentiment_probs']))
        print(orig_accuracy/(si+1), attack_accuracy/(si+1), flipped_pred_rate/(si+1),
            sample_result_dict['gpt_simil'],
            sample_result_dict['attacked_model_simil'],
            sample_result_dict['simil_ratio'],
            sample_result_dict['orig_sentiment_probs'],
            sample_result_dict['attacked_model_sentiment_probs'],
            '\n')
        

    

sim_prompt_embeddings_file.close()

# Convert the result list into a pandas dataframe and write 
df = pd.DataFrame(global_result_list, columns = result_cols)
df.to_csv(output_results_path, index = False)
