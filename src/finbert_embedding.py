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
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification


# Extended from the class implemented in /Users/<user>/Library/Python/3.9/lib/python/site-packages/pytorch_pretrained_bert/modeling.py
class BertForSequenceClassification2(BertForSequenceClassification):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification2, self).__init__(config, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        exp_logits = torch.exp(logits)
        probs = exp_logits/exp_logits.sum()

        return pooled_output, probs






class FinbertEmbedding(object):

    def __init__(self, model_path):

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load pre-trained model (weights)
        self.model = BertForSequenceClassification2.from_pretrained(model_path, 3)

    def process_text(self, text):
        # Tokenize our sentence with the BERT tokenizer
        return ['[CLS]'] + self.tokenizer.tokenize(text)[:510] + ['[SEP]']

    def get_embedding(self,text):

        tokenized_text = self.process_text(text)
        
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            pooled_output, probs = self.model(tokens_tensor)
        
        # Calculate the average of all token embeddings of the last layer before pooling.
        return pooled_output, probs
