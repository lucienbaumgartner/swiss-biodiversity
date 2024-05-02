import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import re

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

def find_nearest_neighbours(target_terms, embeddings_dict, out, n=5):
    # Calculate the similarity matrix
    similarity_matrix, terms = calculate_similarity(embeddings_dict)

    results = {}
    for target_key in target_terms:
        try:
            target_index = terms.index(target_key)
        except ValueError:
            print(f"Target {target_key} not found in embeddings dictionary.")
            continue

        # Extract the similarity scores for the target term
        similarity_scores = similarity_matrix[target_index]

        # Set the similarity of the target term with itself to -1 to exclude it from the results
        similarity_scores[target_index] = -1

        # Get the indices of the top n scores
        nearest_neighbour_indices = np.argsort(similarity_scores)[-n:][::-1]

        results[target_key] = [(terms[index], similarity_scores[index]) for index in nearest_neighbour_indices]

    # Write the results for each target term to the output file
    with open(out, 'w') as file:
        for target_key, neighbours in results.items():
            target_term, target_pos = target_key
            file.write(f'Target Term: {target_term} ({target_pos})\n')
            file.write("Top Nearest Neighbours:\n")
            for neighbour_key, similarity in neighbours:
                neighbour_term, neighbour_pos = neighbour_key
                file.write(f"{neighbour_term} ({neighbour_pos}): {similarity:.4f}\n")
            file.write("\n")

    print(f'Top {n} nearest neighbours for each target term written to {out}')


def create_tsne_plot(target_terms, embeddings_dict, out, radius=10):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = list(embeddings_dict.values())
    terms = list(embeddings_dict.keys())  # terms are now (term, pos) tuples
    embeddings_array = np.array(embeddings)
    tsne_results = tsne.fit_transform(embeddings_array)

    # Find target indices for all terms in the list
    target_indices = []
    for term_pos in target_terms:
        try:
            index = terms.index(term_pos)
            target_indices.append(index)
        except ValueError:
            print(f"Target {term_pos} not found in embeddings dictionary.")

    plt.figure(figsize=(15, 15))
    texts = []
    for i, (term, pos) in enumerate(terms):
        # Check if the point is within radius of any target term
        plot_point = False
        for target_index in target_indices:
            target_coords = tsne_results[target_index]
            distance = np.sqrt(
                (tsne_results[i, 0] - target_coords[0]) ** 2 + (tsne_results[i, 1] - target_coords[1]) ** 2)
            if distance <= radius:
                plot_point = True
                break

        if plot_point:
            color = 'red' if i in target_indices else 'black'
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color)
            label = f"{term} ({pos})"
            texts.append(plt.text(tsne_results[i, 0], tsne_results[i, 1], label, ha='center', va='center', color=color))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.savefig(fname=out, dpi=300, format='png')
    print(f't-SNE plot saved as {out}')


def create_tsne_plot_for_all_files(target_terms, embeddings_dict, out):
    grouped_embeddings = defaultdict(list)
    for (filename, term, pos), embedding in embeddings_dict.items():
        grouped_embeddings[filename].append((term, pos, embedding))

    plt.figure(figsize=(10, 10))
    tsne = TSNE(n_components=2, random_state=42)
    color_map = plt.cm.get_cmap('tab20', len(grouped_embeddings))
    target_coords_by_group = defaultdict(lambda: defaultdict(list))  # Store coords by filename and term-pos pair

    for i, (filename, group_data) in enumerate(grouped_embeddings.items()):
        terms, embeddings = zip(*[(data[0:2], data[2]) for data in group_data])
        embeddings_array = np.array(embeddings)
        tsne_results = tsne.fit_transform(embeddings_array)

        for j, (term, pos) in enumerate(terms):
            if (term, pos) in target_terms:
                target_coords_by_group[filename][(term, pos)].append(
                    (tsne_results[j], filename))  # Save coords and filename

    for filename, targets_coords in target_coords_by_group.items():
        for (term, pos), coords_list in targets_coords.items():
            if len(coords_list) > 1:
                for i in range(len(coords_list) - 1):
                    for j in range(i + 1, len(coords_list)):
                        plt.plot([coords_list[i][0][0], coords_list[j][0][0]],
                                 [coords_list[i][0][1], coords_list[j][0][1]], color='grey', linestyle='--')

            for coords, file_label in coords_list:
                x, y = coords
                plt.scatter(x, y, color='red', s=30)
                plt.annotate(file_label.replace('lemmata_embeddings_', ''), (x, y), textcoords="offset points",
                             xytext=(5, 5), ha='right', color='black', weight='bold', size=14)

    plt.legend()
    plt.savefig(fname=out, dpi=300, format='png')
    plt.close()
    print(f't-SNE plot saved as {out}')

def plot_similarity_heatmap(sim_matrix, keys, out):
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = ax.imshow(sim_matrix, cmap='viridis')

    # Adding labels and ticks
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)

    # Add color bar
    plt.colorbar(heatmap)

    plt.title("Similarity Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(out)
    plt.show()


inputpath = "../output/embeddings/aggregated/type_level/"
files = os.listdir(inputpath)
if ".DS_Store" in files:
    files.remove(".DS_Store")
files.sort()

target_terms = [("biodiversität", "NOUN"), ("biodiversité", "NOUN")]
tt = "biodiversität"
tpos = "NOUN"

for file in files:
    # Load data
    pickle_file = open(inputpath + file, 'rb')
    embeddings_dict = pickle.load(pickle_file)
    file_out = file.replace("embeddings_", "")

    # Write out the nearest neighbour incl. all PoS-tags
    outpath = "../output/similarity_measures/nearest_neighbours/all_pos/"
    out = outpath + tt + '_' + file_out.replace("pkl", "txt")
    find_nearest_neighbours(target_terms=target_terms, embeddings_dict=embeddings_dict, out=out, n=50)

    # Plot the terms in the vicinity of our target term
    outpath = "../output/similarity_measures/t-SNE/all_pos/"
    out = outpath + tt + '_' + file_out.replace("pkl", "png")
    create_tsne_plot(target_terms=target_terms, embeddings_dict=embeddings_dict, out=out, radius=15)

    # Filter out elements where the second key is not 'NN'
    filtered_dict = {key: value for key, value in embeddings_dict.items() if key[1] == tpos}

    # Redo the same for this subset
    outpath = "../output/similarity_measures/nearest_neighbours/nn/"
    out = outpath + tt + '_' + file_out.replace("pkl", "txt")
    find_nearest_neighbours(target_terms=target_terms, embeddings_dict=filtered_dict, out=out, n=50)

    outpath = "../output/similarity_measures/t-SNE/nn/"
    out = outpath + tt + '_' + file_out.replace("pkl", "png")
    create_tsne_plot(target_terms=target_terms, embeddings_dict=filtered_dict, out=out, radius=15)

print("\n\n*** DONE with first part ***\n\n")

# Combine all files
outpath = "../output/similarity_measures/t-SNE/nn/"
out = outpath + tt + '_' + "combinedFiles.png"
combined_dict = load_and_combine_pickles(files)

# Filter out elements where the second key is not 'NN'
filtered_dict = {key: value for key, value in combined_dict.items() if key[2] == tpos}

# Create an overview plot for all files, highlighting the distances between the target term in different docs
create_tsne_plot_for_all_files(target_terms=target_terms, embeddings_dict=combined_dict,out=out)

# Calculate pairwise similarity matrix for target term embeddings across all files
target_dict = {key: value for key, value in filtered_dict.items() if key[1] in [k[0] for k in target_terms]}
sim_matrix, keys = calculate_similarity(target_dict)
write_similarity_to_file(sim_matrix=sim_matrix, keys=keys, key_index=0, filename='../output/similarity_measures/pairwise_similarity_matrix/sim_matrix.txt')
plot_similarity_heatmap(sim_matrix, keys, '../output/similarity_measures/similarity_heatmap.png')