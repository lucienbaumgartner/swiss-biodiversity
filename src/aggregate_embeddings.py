import pickle
import os
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from HanTa import HanoverTagger as ht

# Load the German language model
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

def calculate_word_embeddings(meta_files, embedding_files):

    progress_bar = tqdm(total=len(meta_files), desc="Calculating word embeddings on token and type level", smoothing=1)

    # Make sure file lists have equal length
    assert len(meta_files) == len(embedding_files)

    # Loop over files
    for file_index in range(len(meta_files)):
        # load tokens
        meta_file = open(input_dir + meta_files[file_index],'rb')
        meta = pickle.load(meta_file)

        # load embeddings
        embedding_file = open(input_dir + embedding_files[file_index],'rb')
        embeddings = pickle.load(embedding_file)

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

        # Make sure that word embeddings and the summed word embeddings have same length
        assert len(merged_words) == len(summed_embeddings_by_group)

        embeddings_dict = {term: np.array(embedding) for term, embedding in zip(merged_words, summed_embeddings_list)}

        # Dump the token level data
        output_path = "../output/embeddings/aggregated/token_level/"
        with open(os.path.join(output_path, embedding_files[file_index]), 'wb') as f:
            pickle.dump(embeddings_dict, f)

        lemmata = tagger.tag_sent(merged_words, taglevel=1)

        lemmatized_words, pos_tags = zip(*[(t[1].lower(), t[2]) for t in lemmata])

        # Make sure that lemmatized word embeddings and the summed word embeddings have same length
        assert len(lemmatized_words) == len(summed_embeddings_by_group)

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
embedding_files = [k for k in files if 'embedding' in k]
meta_files = [k for k in files if 'meta' in k]

# aggregate embeddings
calculate_word_embeddings(meta_files=meta_files, embedding_files=embedding_files)

