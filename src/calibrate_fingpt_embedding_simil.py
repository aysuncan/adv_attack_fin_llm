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

import sys
import os
import struct
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from data_loader import load_dataset, subsample_random_balanced_dataset
from fingpt_embedding import FinGPTEmbedding

pd.options.mode.chained_assignment = None  # default='warn'


# Add the arguments and parse them.
parser = argparse.ArgumentParser(description="Calibrate the embedding similarity value function of the FinGPT model on a given dataset.")
parser.add_argument('--dataset_name', type=str, help='A dataset string to choose from the set {"FPB", "TFNS"}', required=True)
parser.add_argument('--no_of_random_sample_pairs', type=int, help='Number of random sample pairs to be used for calibration.', default=2000, required=False)
args = parser.parse_args()


# Input global parameters.
dataset_name = args.dataset_name
no_of_random_sample_pairs_for_calib = args.no_of_random_sample_pairs


# Other global parameters.
model_path = ["../models/fingpt/meta-llama_Llama-2-7b-hf", "../models/fingpt/FinGPT_fingpt-mt_llama2-7b_lora"]
output_calibration_file_path = os.path.join(model_path[1], "embeding_similarity_calibration.txt")
if dataset_name == "FPB":
    in_dataset_path = "../datasets/financial-phrasebank/Sentences_AllAgree.txt"
elif dataset_name == "TFNS":
    in_dataset_path = "../datasets/twitter-financial-news-sentiment/sent_valid.csv"
else:
     raise ValueError('Dataset name ' + dataset_name + ' is not recognized!')









dataset = load_dataset(dataset_name, in_dataset_path)


model = FinGPTEmbedding(model_path)
no_samples = len(dataset)
min_simil = sys.float_info.max
for i in tqdm(range(no_of_random_sample_pairs_for_calib)):

    # Sample a pair of indices.
    s1 = 0
    s2 = 0
    while s1 == s2:
        s1 = random.randint(0, no_samples-1)
        s2 = random.randint(0, no_samples-1)

    text1 = dataset.iloc[s1].text
    text2 = dataset.iloc[s2].text
    
    embedding1, response1 = model.get_embedding(text1)
    embedding2, response2 = model.get_embedding(text2)
    
    simil = np.dot(embedding1, embedding2) + 1 # cosine similarity mapped into [0,2]
    
    if min_simil > simil:
        min_simil = simil
    
    # Print the running min value up until now
    print(min_simil)



# Compute the required linear mapping parameters to map the minimum similarity
# value found to -1 (the minimum possible cosine similarity), while keeping the
# max value (1.0) the same.
#
#   When (x = min_sim) -> ax+b = 0
#   When (x = 2) -> ax+b = 2
#
#   a * min_sim + (2-2a) = 0
#   a * (min_sim-2) = -2
#   a = 2/(2-min_sim)
#   b = 2-2*a
#
simil_scaling = 2.0 / (2.0 - min_simil)
simil_bias = 2.0 - 2.0 * simil_scaling

# Write the calibration results to a file.
with open(output_calibration_file_path, 'w') as f:
    f.write(str(simil_scaling))
    f.write(" ")
    f.write(str(simil_bias))
