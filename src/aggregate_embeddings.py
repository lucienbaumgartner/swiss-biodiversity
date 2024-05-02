import pickle
import os
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re
import spacy
from spacy.tokens import Doc

"""
from spacy_lefff import LefffLemmatizer
from spacy.language import Language

@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer()
"""

# Load the German language model
germanTagger = spacy.load('de_core_news_sm')

# Load the French language model
frenchTagger = spacy.load('fr_core_news_sm')
#frenchTagger.add_pipe('french_lemmatizer', name='lefff')

def calculate_word_embeddings(meta_files, embedding_files):

    progress_bar = tqdm(total=len(meta_files), desc="Calculating word embeddings on token and type level", smoothing=1)

    # Make sure file lists have equal length
    assert len(meta_files) == len(embedding_files)

    # Loop over files
    for file_index in range(len(meta_files)):
        print(meta_files[file_index])
        # load tokens
        meta_file = open(input_dir + meta_files[file_index],'rb')
        meta = pickle.load(meta_file)

        # load embeddings
        embedding_file = open(input_dir + embedding_files[file_index],'rb')
        embeddings = pickle.load(embedding_file)

        # get the language for the analysis
        lang = re.search(r"_(de|fr)_", meta_files[file_index]).group(1)
        print(lang)

        # Check that tokens and embeddings have same length
        assert len(meta) == len(embeddings)

        # Annotate groups of word part tokens that need to be aggregated on word level
        group_indices = []
        current_index = 1

        for token in meta:
            if not token.startswith("##"):
                if group_indices:  # Increment index if it's not the first token
                    current_index += 1
            group_indices.append(current_index)

        # Merge word part tokens to full word tokens based on group indices
        grouped_tokens = defaultdict(list)
        for token, index in zip(meta, group_indices):
            grouped_tokens[index].append(token.lstrip('##'))

        merged_words = [''.join(group) for group in grouped_tokens.values()]

        # Initialize a dictionary to sum embeddings by group, using a zero tensor of the correct size
        summed_embeddings_by_group = defaultdict(lambda: torch.zeros(768))

        # Sum embeddings by their group index
        for embedding, index in zip(embeddings, group_indices):
            summed_embeddings_by_group[index] += embedding  # Direct addition of tensors

        # Convert the summed embeddings to a list if necessary
        summed_embeddings_list = list(summed_embeddings_by_group.values())

        # Filtering out pairs where the string is an empty string
        filtered_pairs = [(s, t) for s, t in zip(merged_words, summed_embeddings_list) if s != ""]

        # Unzipping filtered pairs back into separate lists
        merged_words, summed_embeddings_list = zip(*filtered_pairs) if filtered_pairs else ([], [])

        # Make sure that word embeddings and the summed word embeddings have same length
        assert len(merged_words) == len(summed_embeddings_list)

        embeddings_dict = {term: np.array(embedding) for term, embedding in zip(merged_words, summed_embeddings_list)}

        # Dump the token level data
        output_path = "../output/embeddings/aggregated/token_level/"
        with open(os.path.join(output_path, embedding_files[file_index]), 'wb') as f:
            pickle.dump(embeddings_dict, f)

        if lang == "de":
            #merged_words = list(map(lambda x: x.replace('[UNK]', "'"), merged_words))
            doc = Doc(germanTagger.vocab, merged_words)
            annotated_doc = germanTagger(doc)
            lemmatized_words, pos_tags = zip(*[(t.lemma_, t.pos_) for t in annotated_doc])
            lemmatized_words = [t.lower() for t in lemmatized_words]

        elif lang == "fr":
            merged_words = list(map(lambda x: x.replace('[UNK]', "'"), merged_words))
            doc = Doc(frenchTagger.vocab, merged_words)
            annotated_doc = frenchTagger(doc)
            lemmatized_words, pos_tags = zip(*[(t.lemma_, t.pos_) for t in annotated_doc])
            lemmatized_words = [t.lower() for t in lemmatized_words]

        elif lang is None:
            print(f"Unknown language; aborting: {meta_files[file_index]}")

        # Make sure that lemmatized word embeddings and the summed word embeddings have same length
        assert len(lemmatized_words) == len(summed_embeddings_list)

        embeddings_dict = {term: {"embeddings": np.array(embedding), "pos": pos} for term, embedding, pos in zip(lemmatized_words, summed_embeddings_list, pos_tags)}

        # Step 1: Initialize a new dictionary to collect embeddings by (term, pos) pairs
        embeddings_by_term_pos = defaultdict(list)

        # Step 2: Collect embeddings by (term, pos) pairs
        for term, data in embeddings_dict.items():
            term_pos_key = (term, data['pos'])  # Create a composite key
            embeddings_by_term_pos[term_pos_key].append(data['embeddings'])

        # Step 3: Sum embeddings for each (term, pos) pair
        summed_embeddings_dict = {term_pos: np.sum(np.array(embeddings), axis=0) for term_pos, embeddings in
                                  embeddings_by_term_pos.items()}

        tags_to_remove = ['$(', '$,', '$.', 'UNKNOWN', 'XY']
        summed_embeddings_dict = {key: value for key, value in summed_embeddings_dict.items() if key[1] not in tags_to_remove}

        # Dump the type level data
        output_path = "../output/embeddings/aggregated/type_level/lemmata_"
        with open(output_path + embedding_files[file_index], 'wb') as f:
            pickle.dump(summed_embeddings_dict, f)

        progress_bar.update(1)

    progress_bar.close()

# load data
input_dir = "../output/embeddings/raw/"
files = os.listdir(input_dir)
if ".DS_Store" in files:
    files.remove(".DS_Store")
files.sort()
print(files)
embedding_files = [k for k in files if 'embedding' in k]
meta_files = [k for k in files if 'meta' in k]

assert len(embedding_files) == len(meta_files)

# aggregate embeddings
calculate_word_embeddings(meta_files=meta_files, embedding_files=embedding_files)

