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
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0

# Adapted from https://github.com/AI4Finance-Foundation/FinGPT/blob/master/FinGPT_Inference_Llama2_13B_falcon_7B_for_Beginners.ipynb

class FinGPTEmbedding(object):

    def __init__(self, model_path):
        self.template_dict = {'default': 'Instruction: {instruction}\nInput: {input}\nAnswer: '}
        self.tasks = ['Financial Sentiment Analysis', 'Financial Relation Extraction', 'Financial Headline Classification', 'Financial Named Entity Recognition']
        self.instructions = ['What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.',
                                'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.',
                                'Does the news headline talk about price going up? Please choose an answer from {Yes/No}.',
                                'Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.']
        self._default_task_id = 0
        self.load_model(model_path[0], model_path[1])

            
    def load_model(self, base_model, peft_model):
        model = AutoModelForCausalLM.from_pretrained(
                base_model, trust_remote_code=True,
                device_map="auto")
        model.model_parallel = True
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, peft_model)
        self.model = model.eval()
        self.tokenizer = tokenizer


    def get_embedding(self, text):
        prompt = 'Instruction: {instruction}\nInput: {input}\nAnswer: '.format(input=text, instruction=self.instructions[self._default_task_id])
        tokenized_inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, return_token_type_ids=False)
        dict_inputs = {key: value.to(self.model.device) for key, value in tokenized_inputs.items()}

        with torch.no_grad():
            tokenized_response = self.model.generate(**dict_inputs, max_length=512, do_sample=False,eos_token_id=self.tokenizer.eos_token_id)
            
            # For embeddings, we use the "prompt-based last token" approach "Scaling Sentence Embeddings with Large Language Models".
            # Note that since our output is a single word for most tasks, we don't need to specify/ask that in the prompt.
            last_hidden_state = self.model(**dict_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
            
        output = self.tokenizer.decode(tokenized_response[0], skip_special_tokens=True)

        idx_of_the_last_non_padding_token = tokenized_inputs.attention_mask.bool().sum(1)-1
        embedding = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token]
        embedding = embedding[0].cpu().detach().numpy()
        embedding = embedding / np.linalg.norm(embedding) # make it a unit vector
        
        # Get the answer and strip the rest.
        ans = '\nAnswer: '
        output = output[(output.rfind(ans) + len(ans)):]
        if 'neutral' in output:
            pred = 2
        elif 'negative' in output:
            pred = 1
        elif 'positive' in output:
            pred = 0
        else:
            pred = -1
        
        return embedding, pred
