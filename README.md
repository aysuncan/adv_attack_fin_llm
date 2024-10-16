# Prerequisites

## 1. Model Installation

Models are not shared with this repository because they are copyright-protected and not viable for third party distribution. You can obtain them from the links below.

### FinBERT
Download the FinBERT model from [here](https://huggingface.co/ProsusAI/finbert) and place all the files in the directory "../models/finbert"



### FinGPT

Install the Llama-2-7B model from [here](https://huggingface.co/meta-llama/Llama-2-7b-hf) and place all the files in the directory "../models/fingpt/meta-llama_Llama-2-7b-hf"



Install the FinGPT model from [here](https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora) and place all the files in the directory "../models/fingpt/FinGPT_fingpt-mt_llama2-7b_lora"

## 2. Datasets

Download the following datasets and place their folders under the datasets directory of this repository.


Download the financial-phrasebank (FPB) dataset from [here](https://huggingface.co/datasets/takala/financial_phrasebank/tree/main/data), unzip the archive and rename the resulting fodler to 'financial-phrasebank'.


Clone the SEntFiN repository from [here](https://github.com/pyRis/SEntFiN) and make sure that the folder name is 'SEntFiN'.

Download the twitter-financial-news-sentiment (TFNS) dataset from [here](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/tree/main) and make sure that the folder name is 'twitter-financial-news-sentiment'.

## 3. Requirements
The repository is developed with Python 3.11.5. All the dependencies can be installed with

```console
pip install -r requirements.txt
```

# Reproducing Results

## 1. Create Paraphrases and Their Embeddings for a Dataset

ATTENTION: This and the following steps overwrite the previous results, so be careful when running them. If your goal is to create new results without overwriting the old ones, modify the *version_text* parameter inside the three scripts called below.

```console
cd src
python create_paraphrases_and_embeddings.py --openai <key_string> --dataset_name <dataset_string>
```

* Replace \<key_string\> with your OpenAI key string.
* Replace \<dataset_string\> with one of the following strings: "FPB", "TFNS", "SEntFiN" depending on which datasets you want to use.



## 2. Predict the Financial Sentiment for All the Samples For a Dataset Using the Attacker Model (GPT-4o)

```console
cd src
python predict_sentiments_with_attacker.py --openai <key_string> --dataset_name <dataset_string>
```


## 3. Perform the Adversarial Attack on One of the Models and Generate the Final Results

### Attacking FinBERT
```console
cd src
python attack_finbert.py --openai <key_string> --dataset_name <dataset_string>
```

### Attacking FinGPT --> REQUIRES A GPU WITH LARGE ENOUGH MEMORY!
```console
cd src
python attack_fingpt.py --openai <key_string> --dataset_name <dataset_string>
```

Results are created in the respective dataset directory under "../results/". The final results of the attack are contained in the csv file prefixed "performance_results".

# Citation
If you find our work useful please cite:

```
@mastersthesis{zora262354,
  author = {Can Turetken, Aysun},
  title = {An Adversarial Attack Approach on Financial LLMs Driven by Embedding-Similarity Optimization},
  school = {University of Zurich},
  month = {July},
  year = {2024},
  url = {https://doi.org/10.5167/uzh-262354}
}
```

```
@misc{ssrnpaper2024,
  title={Battle of Transformers: Adversarial Attacks on Financial Sentiment Models},
  author={Can Turetken, Aysun and Leippold, Markus},
  year={2024},
  note={SSRN Working Paper},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4977483}
}
```
