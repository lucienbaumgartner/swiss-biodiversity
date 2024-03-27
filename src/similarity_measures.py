import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os

def load_and_combine_pickles(file_paths):
    combined_dict = {}
    for file_path in file_paths:
        with open(inputpath + file_path, 'rb') as f:
            # Load the dictionary from the current file
            current_dict = pickle.load(f)

            # Extract filename without extension to use as an additional key
            filename = os.path.splitext(os.path.basename(file_path))[0]

            # Modify the current dictionary to include the filename in its keys
            modified_dict = {((filename,) + key): value for key, value in current_dict.items()}

            # Merge the modified dictionary with the combined dictionary
            combined_dict.update(modified_dict)

    return combined_dict

def write_similarity_to_file(sim_matrix, keys, key_index, filename):
    group_keys = [t[key_index] for t in keys]
    with open(filename, 'w') as f:
        # Write terms as the first line (header)
        f.write('\t'.join(group_keys) + '\n')
        # Write each row of the similarity matrix
        for i, row in enumerate(sim_matrix):
            row_data = '\t'.join([str(sim) for sim in row])
            f.write(f"{group_keys[i]}\t{row_data}\n")

def calculate_similarity(embeddings_dict):
    # embeddings_dict is expected to have (term, pos) tuples as keys
    embeddings = list(embeddings_dict.values())
    terms = list(embeddings_dict.keys())  # Terms are now (term, pos) tuples
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix, terms

def find_nearest_neighbours(target_term, target_pos, embeddings_dict, out, n=5):
    # Calculate the similarity matrix
    similarity_matrix, terms = calculate_similarity(embeddings_dict)
    target_key = (target_term, target_pos)
    try:
        target_index = terms.index(target_key)
    except ValueError:
        print(f"Target ({target_term}, {target_pos}) not found in embeddings dictionary.")
        return

    # Extract the similarity scores for the target term
    similarity_scores = similarity_matrix[target_index]

    # Set the similarity of the target term with itself to -1 to exclude it from the results
    similarity_scores[target_index] = -1

    # Get the indices of the top n scores
    nearest_neighbour_indices = np.argsort(similarity_scores)[-n:][::-1]

    # Write the top n nearest neighbours to the output file
    with open(out, 'w') as file:
        file.write(f'Target Term: {target_term} ({target_pos})\n')
        file.write("Top Nearest Neighbours:\n")
        for index in nearest_neighbour_indices:
            neighbour_key = terms[index]
            neighbour_term, neighbour_pos = neighbour_key
            similarity = similarity_scores[index]
            file.write(f"{neighbour_term} ({neighbour_pos}): {similarity:.4f}\n")

    print(f'Top {n} nearest neighbours written to {out}')

def create_tsne_plot(target_term, target_pos, embeddings_dict, out, radius=10):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = list(embeddings_dict.values())
    terms = list(embeddings_dict.keys())  # terms are now (term, pos) tuples
    embeddings_array = np.array(embeddings)
    tsne_results = tsne.fit_transform(embeddings_array)

    target_key = (target_term, target_pos)
    try:
        target_index = terms.index(target_key)
    except ValueError:
        print(f"Target ({target_term}, {target_pos}) not found in embeddings dictionary.")
        return

    target_coords = tsne_results[target_index]

    # Calculate distances from the target term to all other points
    distances = np.sqrt((tsne_results[:, 0] - target_coords[0])**2 + (tsne_results[:, 1] - target_coords[1])**2)

    plt.figure(figsize=(10, 10))
    texts = []
    for i, (term, pos) in enumerate(terms):
        # Only plot points within the defined radius from the target term
        if distances[i] <= radius:
            color = 'red' if (term, pos) == target_key else 'black'
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color)
            label = f"{term} ({pos})"
            texts.append(plt.text(tsne_results[i, 0], tsne_results[i, 1], label, ha='center', va='center', color=color))

    # Use adjust_text to reduce overlaps, passing the list of text objects and corresponding points
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.savefig(fname=out, dpi=300, format='png')
    print(f't-SNE plot saved as {out}')

def create_tsne_plot_for_all_files(target_term, target_pos, embeddings_dict, out):
    grouped_embeddings = defaultdict(list)
    for (filename, term, pos), embedding in embeddings_dict.items():
        grouped_embeddings[filename].append((term, pos, embedding))

    plt.figure(figsize=(10, 10))
    tsne = TSNE(n_components=2, random_state=42)
    color_map = plt.cm.get_cmap('tab20', len(grouped_embeddings))
    target_coords_by_group = defaultdict(list)  # To store target term coordinates by group

    # Compute and plot t-SNE for each group separately
    for i, (filename, group_data) in enumerate(grouped_embeddings.items()):
        terms, embeddings = zip(*[(data[0:2], data[2]) for data in group_data])
        embeddings_array = np.array(embeddings)
        tsne_results = tsne.fit_transform(embeddings_array)

        # Check for target term in this group and plot
        for j, (term, pos) in enumerate(terms):
            if (term, pos) == (target_term, target_pos):
                target_coords_by_group[filename].append((tsne_results[j], filename))  # Save target term coords and filename
            plt.scatter(tsne_results[j, 0], tsne_results[j, 1], color=color_map(i), label=filename if j == 0 else "", alpha=0.5, s=10)

    if len(target_coords_by_group) > 1:  # Draw lines if target term is in more than one group
        target_coords_all = [(coord, file_label) for coords_list in target_coords_by_group.values() for coord, file_label in coords_list]
        for i in range(len(target_coords_all) - 1):
            for j in range(i + 1, len(target_coords_all)):
                plt.plot([target_coords_all[i][0][0], target_coords_all[j][0][0]], [target_coords_all[i][0][1], target_coords_all[j][0][1]], color='grey', linestyle='--')

    # Highlight the target term in red, draw lines, and annotate
    for filename, coords_list in target_coords_by_group.items():
        for coords, file_label in coords_list:
            x, y = coords
            plt.scatter(x, y, color='red', s=30)
            plt.annotate(file_label.replace('lemmata_embeddings_', ''), (x, y), textcoords="offset points", xytext=(5,5), ha='right', color='black', weight='bold', size = 14)

    plt.legend()
    plt.savefig(fname=out, dpi=300, format='png')
    plt.close()  # Close the plot to free up memory
    print(f't-SNE plot saved as {out}')


inputpath = "../output/embeddings/aggregated/type_level/"
files = os.listdir(inputpath)
if ".DS_Store" in files:
    files.remove(".DS_Store")
files.sort()

target_term = "biodiversität"
target_pos = "NN"

for file in files:
    # Load data
    pickle_file = open(inputpath + file, 'rb')
    embeddings_dict = pickle.load(pickle_file)
    file_out = file.replace("embeddings_", "")

    # Write out the nearest neighbour incl. all PoS-tags
    outpath = "../output/similarity_measures/nearest_neighbours/all_pos/"
    out = outpath + target_term + '_' + file_out.replace("pkl", "txt")
    find_nearest_neighbours(target_term=target_term, target_pos=target_pos, embeddings_dict=embeddings_dict, out=out, n=50)

    # Plot the terms in the vicinity of our target term
    outpath = "../output/similarity_measures/t-SNE/all_pos/"
    out = outpath + target_term + '_' + file_out.replace("pkl", "png")
    create_tsne_plot(target_term=target_term, target_pos=target_pos, embeddings_dict=embeddings_dict, out=out, radius=15)

    # Filter out elements where the second key is not 'NN'
    filtered_dict = {key: value for key, value in embeddings_dict.items() if key[1] == 'NN'}

    # Redo the same for this subset
    outpath = "../output/similarity_measures/nearest_neighbours/nn/"
    out = outpath + target_term + '_' + file_out.replace("pkl", "txt")
    find_nearest_neighbours(target_term=target_term, target_pos=target_pos, embeddings_dict=filtered_dict, out=out, n=50)

    outpath = "../output/similarity_measures/t-SNE/nn/"
    out = outpath + target_term + '_' + file_out.replace("pkl", "png")
    create_tsne_plot(target_term=target_term, target_pos=target_pos, embeddings_dict=filtered_dict, out=out, radius=15)

# Combine all files
outpath = "../output/similarity_measures/t-SNE/nn/"
out = outpath + target_term + '_' + "combinedFiles.png"
combined_dict = load_and_combine_pickles(files)

# Filter out elements where the second key is not 'NN'
filtered_dict = {key: value for key, value in combined_dict.items() if key[2] == 'NN'}

# Create an overview plot for all files, highlighting the distances between the target term in different docs
create_tsne_plot_for_all_files(target_term=target_term,target_pos=target_pos, embeddings_dict=combined_dict,out=out)

# Calculate pairwise similarity matrix for target term embeddings across all files
target_dict = {key: value for key, value in filtered_dict.items() if key[1] == target_term}

sim_matrix, keys = calculate_similarity(target_dict)

write_similarity_to_file(sim_matrix=sim_matrix, keys=keys, key_index=0, filename='../output/similarity_measures/pairwise_similarity_matrix/sim_matrix.txt')