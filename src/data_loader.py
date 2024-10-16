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

import pandas as pd
import ast

def load_FPB_dataset(filepath):
	finbert_label_dict = {'positive\n': 0, 'negative\n': 1, 'neutral\n': 2}
	data_list = []
	with open(filepath, encoding="iso-8859-1") as f:
		for id_, line in enumerate(f):
			text, gt = line.rsplit("@", 1)
			label = finbert_label_dict[gt]
			data_list.append((text, label))
	df = pd.DataFrame(data_list, columns =['text', 'label'])
	return df


def load_SEntFiN_dataset(filepath):
    df = pd.read_csv(filepath)
    # Function to safely evaluate dictionary strings
    def safe_eval(decision_str):
        try:
            return ast.literal_eval(decision_str)
        except (ValueError, SyntaxError):
            return None
    # Apply the safe_eval function to the 'Decisions' column
    df['Decisions'] = df['Decisions'].apply(safe_eval)

    # Create new DataFrame with only the columns title (text) and sentiment (integer)
    new_rows = []
    for i, row in df.iterrows():
        decisions = row['Decisions']
        if decisions and (len(decisions) ==1 or all(value == list(decisions.values())[0] for value in decisions.values())):
            sentiments = list(decisions.values())
            new_rows.append({'text': row['Title'],'label': sentiments[0]})

    new_df = pd.DataFrame(new_rows)
    label_dict = {'positive':0,'negative':1,'neutral':2}
    new_df['label'] = new_df['label'].map(label_dict)
    return new_df

def load_dataset(dataset_name, dataset_path):
    # Load the financial sentiment analysis dataset
    if dataset_name == "FPB":
        dataset = load_FPB_dataset(dataset_path)
    elif dataset_name == "TFNS":
        # Twitter-financial-news-sentiment.
        dataset = pd.read_csv(dataset_path)

        # Make the dataset labels the same as that of the Finbert model loaded.
        dataset['label'] = dataset['label'].replace([0],-1)
        dataset['label'] = dataset['label'].replace([1],0)
        dataset['label'] = dataset['label'].replace([-1],1)
    elif dataset_name == "SEntFiN":
        dataset = load_SEntFiN_dataset(dataset_path)
    else:
        raise ValueError('Dataset name ' + dataset_name + ' is not recognized!')
        
    return dataset


def subsample_random_balanced_dataset(input_df, no_samples_per_class):
    random_seed = 345
    shuffled_dataset = input_df.sample(frac=1, random_state=random_seed).reset_index(drop=True) # randomly shuffle dataset samples
    grouped = shuffled_dataset.groupby('label')
    sampled = grouped.apply(lambda x:x.sample(n=min(len(x), no_samples_per_class), random_state=random_seed))
    sampled = sampled.reset_index(drop=True)
    return sampled
