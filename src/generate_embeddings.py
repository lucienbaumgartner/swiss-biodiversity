import os
import csv
import re
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from operator import itemgetter
from collections import OrderedDict
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pickle

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)

def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(indexed_tokens)
  # convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])
  return tokenized_text, tokens_tensor, segments_tensor

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)
    # intialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
    # "token" is a [12 x 768] tensor
    # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        """
        As an alternative to the sum of the last four layers, one can also concenate the last four layers or extract the second to last layer
        """
        #sum_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        #sum_vec = token[-2]
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

def wrap_embeddings(sentences):
    progress_bar = tqdm(total=len(sentences), desc="Generating Embeddings", smoothing=1)
    context_embeddings = []
    context_tokens = []
    for sentence in sentences:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
        try:
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        except RuntimeError as e:
            continue
        # make ordered dictionary to keep track of the position of each   word
        tokens = OrderedDict()
        # loop over tokens in sensitive sentence
        for token in tokenized_text[1:-1]:
            # keep track of position of word and whether it occurs multiple times
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

            # compute the position of the current token
            token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
            current_index = token_indices[tokens[token]-1]
            # get the corresponding embedding
            token_vec = list_token_embeddings[current_index]

            # save values
            context_tokens.append(token)
            context_embeddings.append(token_vec)
        # add delimiter
        context_tokens.append("@@@")
        context_embeddings.append(torch.tensor([0]).repeat(768))
        progress_bar.update(1)
    progress_bar.close()
    return(context_embeddings, context_tokens)

input_folder_path = "../input/clean/"
txt_files = os.listdir(input_folder_path)
if ".DS_Store" in txt_files:
    txt_files.remove(".DS_Store")

print("Files to process:")
for txt_file in txt_files:
    print(txt_file)

for txt_file in txt_files:
    with open(input_folder_path + txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    lang = re.search(r"_(de|fr)_", txt_file).group(1)
    if lang == "de":
        lang = "german"
    if lang == "fr":
        lang = "french"
    sentences = sent_tokenize(text, language=lang)
    sentences = [sentence.replace('\n', '') for sentence in sentences]
    sentences = [sentence.replace(' -', '-') for sentence in sentences]
    sentences = [' '.join(re.split(r'\s+', sentence)) for sentence in sentences]
    context_embeddings, context_tokens = wrap_embeddings(sentences)

    filepath = os.path.join('../output/embeddings/raw/')
    name = 'metadata_' + txt_file.replace("txt", "pkl")
    with open(os.path.join(filepath, name), 'wb') as f:
        pickle.dump(context_tokens, f)

    name = 'embeddings_' + txt_file.replace("txt", "pkl")
    with open(os.path.join(filepath, name), 'wb') as f:
        pickle.dump(context_embeddings, f)
