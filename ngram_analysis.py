import torch
import json
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from scipy.spatial import procrustes
from rouge_score import rouge_scorer
from typing import Tuple
from matplotlib import pyplot as plt
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  # for tf-idf vectors
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import itertools
import pprint
from matplotlib.ticker import PercentFormatter
import seaborn as sns


pp = pprint.PrettyPrinter(indent=4)
scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

# Ensure you have the required data files
nltk.download('punkt')

# creates from truthfulqa
def create_dataset(file_path, first=False):
    dataset ={"question": [], "right_answer": [], "halu_answer": []}
    data = pd.read_csv(file_path)

    for i, entry in data.iterrows():
        dataset["question"].append(entry["Question"])
        tmp = []
        for answer in entry["Correct Answers"].split(';'):
            tmp.append(answer)
            if first:
                break
        dataset["right_answer"].append(tmp)
        tmp = []
        for answer in entry["Incorrect Answers"].split(';'):
            tmp.append(answer)
            if first:
                break
        dataset["halu_answer"].append(tmp)
    return dataset

def randomize_dataset(dataset):
    randomized_dataset = {"question": [], "answer": [], "label": []}
    for i in range(len(dataset["question"])):
        for j in range(len(dataset["right_answer"][i])):
            randomized_dataset["question"].append(dataset["question"][i])
            randomized_dataset["answer"].append(dataset["right_answer"][i][j])
            randomized_dataset["label"].append(1)
        for j in range(len(dataset["halu_answer"][i])):
            randomized_dataset["question"].append(dataset["question"][i])
            randomized_dataset["answer"].append(dataset["halu_answer"][i][j])
            randomized_dataset["label"].append(0)

    return randomized_dataset

def create_global_vocab(dataset):
    vocab = {}
    for i in range(len(dataset["question"])):
        words = nltk.word_tokenize(dataset["question"][i])
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
        words = nltk.word_tokenize(dataset["answer"][i])
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

# Create unified vocabulary from real and hallucinated answers
def create_local_vocab(real_answers, halu_answers, ngram):
    vocab = {}
    all_texts = real_answers + halu_answers
    for text in all_texts:
        # Generate n-grams from tokenized words
        grams = list(ngrams(word_tokenize(text), ngram))
        for gram in grams:
            # Join the tuple into a single string
            gram_str = ' '.join(gram)
            if gram_str not in vocab:
                vocab[gram_str] = len(vocab)
    return vocab

def calculate_svd(real, halu):
    svd = TruncatedSVD(n_components=10, random_state=42)
    real_svd = svd.fit_transform(X_real)
    halu_svd = svd.fit_transform(X_hallucinated)

    return real_svd, halu_svd

def calculate_pairwise_distribution(dataset):
    # Step 1: Group the dataset by question
    grouped_data = defaultdict(lambda: {"right_answers": [], "halu_answers": []})

    for q, ra, ha in zip(dataset["question"], dataset["right_answer"], dataset["halu_answer"]):
        grouped_data[q]["right_answers"].append(ra)
        grouped_data[q]["halu_answers"].append(ha)

    # Step 2: Initialize lists to store the results
    procrustes_distances = {
        "right_right": [],
        "halu_halu": [],
        "right_halu": []
    }

    # Step 3: Iterate through each group and compute distances
    for question, answers in grouped_data.items():
        right_answers = answers["right_answers"]
        halu_answers = answers["halu_answers"]
        
        # Compute right_answer to right_answer distances
        if len(right_answers) > 1:
            for ra1, ra2 in itertools.combinations(right_answers, 2):
                if not np.array_equal(ra1, ra2):
                    mtx1, mtx2, disparity = procrustes(ra1.reshape(1, -1), ra2.reshape(1, -1))
                    procrustes_distances["right_right"].append({
                        "question": question,
                        "answer_pair": ("right_answer_1", "right_answer_2"),
                        "distance": disparity
                    })
        
        # Compute halu_answer to halu_answer distances
        if len(halu_answers) > 1:
            for ha1, ha2 in itertools.combinations(halu_answers, 2):
                if not np.array_equal(ha1, ha2):
                    mtx1, mtx2, disparity = procrustes(ha1.reshape(1, -1), ha2.reshape(1, -1))
                    procrustes_distances["halu_halu"].append({
                        "question": question,
                        "answer_pair": ("halu_answer_1", "halu_answer_2"),
                        "distance": disparity
                    })
        
        # Compute right_answer to halu_answer distances
        for ra in right_answers:
            for ha in halu_answers:
                if not np.array_equal(ra, ha):
                    mtx1, mtx2, disparity = procrustes(ra.reshape(1, -1), ha.reshape(1, -1))
                    procrustes_distances["right_halu"].append({
                        "question": question,
                        "answer_pair": ("right_answer", "halu_answer"),
                        "distance": disparity
                    })

# Create input for SVD
def prepare_svd_input(grams, vocab, ngram):
    input_matrix = []
    for i in range(ngram):
        hash_map = [0] * len(vocab)
        for gram in grams:
            print(gram[i])
            if gram[i] in vocab:
                hash_map[vocab[gram[i]]] += 1
        input_matrix.append(hash_map)

    return np.array(input_matrix).T  # A matrix of shape (len(vocab), ngram)


# Create a unified n-gram matrix
def create_unified_ngram_matrix(real_texts, halu_texts, ngram_range=(1, 1)):
    """
    Creates position-wise n-gram matrices for real and halu texts.
    
    Parameters:
    - real_texts (List[str]): List of real answer sentences.
    - halu_texts (List[str]): List of halu (hallucinated) answer sentences.
    - ngram_range (Tuple[int, int], optional): The lower and upper boundary of the range of n-values for different n-grams to be extracted. Defaults to (1, 1).
    
    Returns:
    - real_matrix (numpy.ndarray): Matrix of shape (n, vocab_size) representing word occurrences in each n-gram position for real_texts.
    - halu_matrix (numpy.ndarray): Matrix of shape (n, vocab_size) representing word occurrences in each n-gram position for halu_texts.
    """
    min_n, max_n = ngram_range
    assert min_n == max_n, "This function currently supports fixed n-gram sizes (min_n must equal max_n)."
    n = min_n  # Size of the n-gram
    batch_size_real = len(real_texts)
    batch_size_halu = len(halu_texts)
    
    # Combine both lists to create a unified vocabulary
    combined_texts = real_texts + halu_texts
    
    # Initialize CountVectorizer to build the vocabulary
    # Using token_pattern to include words with apostrophes and hyphens
    vectorizer = CountVectorizer(ngram_range=(1,1), token_pattern=r"(?u)\b\w+\b")
    
    # Fit the vectorizer on the combined texts to build the vocabulary
    vectorizer.fit(combined_texts)
    
    # Retrieve the vocabulary and its size
    vocabulary = vectorizer.vocabulary_
    vocab_size = len(vocabulary)
    
    # Initialize word to index mapping for quick lookup
    word_to_index = vocabulary
    
    # Initialize the matrices: n positions x vocab_size
    # Each row corresponds to a position in the n-gram (0-based indexing)
    # Each column corresponds to a unique word from the vocabulary
    real_matrix = np.zeros((batch_size_real, n, vocab_size), dtype=int)
    halu_matrix = np.zeros((batch_size_halu, n, vocab_size), dtype=int)

    
    def extract_ngrams(text, n):
        """
        Extracts n-grams from a given text.
        
        Parameters:
        - text (str): The input sentence.
        - n (int): The size of the n-gram.
        
        Returns:
        - List[List[str]]: A list of n-grams, each represented as a list of words.
        """
        tokens = text.lower().split()
        if len(tokens) < n:
            return []
        return [tokens[i:i+n] for i in range(len(tokens)-n+1)]
    
    def process_texts(texts, matrix):
        """
        Processes a list of texts and updates the corresponding matrix.
        
        Parameters:
        - texts (List[str]): List of sentences to process.
        - matrix (numpy.ndarray): Matrix to update with word counts.
        
        Returns:
        - numpy.ndarray: Updated matrix with word counts.
        """
        for i, text in enumerate(texts):
            ngrams = extract_ngrams(text, n)
            for gram in ngrams:
                for position, word in enumerate(gram):
                    if word in word_to_index:
                        idx = word_to_index[word]
                        matrix[i, position, idx] +=1
        return matrix
    
    # Process real_texts and halu_texts to populate the matrices
    real_matrix = process_texts(real_texts, real_matrix)
    halu_matrix = process_texts(halu_texts, halu_matrix)
  
    return real_matrix, halu_matrix, vocabulary

def create_cooccurrence_matrix(texts, vocab, ngram_range=(1, 1)):
    """
    Creates a co-occurrence matrix for the given texts and vocabulary.
    
    Parameters:
    - texts (List[str]): List of sentences to process.
    - vocab (Dict[str, int]): Vocabulary with word to index mapping.
    - ngram_range (Tuple[int, int], optional): The lower and upper boundary of the range of n-values for different n-grams to be extracted. Defaults to (1, 1).
    
    Returns:
    - cooccurrence_matrix (numpy.ndarray): Co-occurrence matrix of shape (vocab_size, vocab_size).
    """
    min_n, max_n = ngram_range
    assert min_n == max_n, "This function currently supports fixed n-gram sizes (min_n must equal max_n)."
    n = min_n  # Size of the n-gram
    batch_size = len(texts)
    
    # Initialize the co-occurrence matrix
    vocab_size = len(vocab)
    cooccurrence_matrix = np.zeros((batch_size, vocab_size, vocab_size), dtype=int)
    
    def extract_ngrams(text, n):
        """
        Extracts n-grams from a given text.
        
        Parameters:
        - text (str): The input sentence.
        - n (int): The size of the n-gram.
        
        Returns:
        - List[List[str]]: A list of n-grams, each represented as a list of words.
        """
        tokens = text.lower().split()
        if len(tokens) < n:
            return []
        return [tokens[i:i+n] for i in range(len(tokens)-n+1)]
    
    def process_texts(texts, matrix):
        """
        Processes a list of texts and updates the co-occurrence matrix.
        
        Parameters:
        - texts (List[str]): List of sentences to process.
        - matrix (numpy.ndarray): Co-occurrence matrix to update.
        
        Returns:
        - numpy.ndarray: Updated co-occurrence matrix.
        """
        for i, text in enumerate(texts):
            ngrams = extract_ngrams(text, n)
            for gram in ngrams:
                word1 = gram[0]
                if word1 in vocab:
                    idx1 = vocab[word1]
                    word2 = gram[1]
                    if word2 in vocab:
                        idx2 = vocab[word2]
                        matrix[i, idx1, idx2] += 1
        return matrix
    
    # Process the texts to populate the co-occurrence matrix
    cooccurrence_matrix = process_texts(texts, cooccurrence_matrix)
    return cooccurrence_matrix

# def create_unified_ngram_matrix(real_texts, halu_texts, ngram_range):
#     # Combine both lists to create a unified vocabulary
#     combined_texts = real_texts + halu_texts

#     # Initialize CountVectorizer with the specified ngram_range
#     vectorizer = CountVectorizer(ngram_range=ngram_range)

#     # Fit the vectorizer on the combined texts
#     vectorizer.fit(combined_texts)

#     # Transform real_texts and halu_texts separately
#     real_matrix = vectorizer.transform(real_texts)
#     halu_matrix = vectorizer.transform(halu_texts)

#     # Optionally, convert sparse matrices to dense if needed
#     # real_matrix = real_matrix.toarray()
#     # halu_matrix = halu_matrix.toarray()

#     # Retrieve the vocabulary
#     vocabulary = vectorizer.vocabulary_

#     return real_matrix, halu_matrix, vocabulary

# Function to pad lists to the same length
def pad_lists(data):
    max_length = max(len(data[key]) for key in data)
    for key in data:
        current_length = len(data[key])
        if current_length < max_length:
            data[key].extend([None] * (max_length - current_length))
    return data

def plot_probability_distribution_histogram(dataset):
    # Separate real-real, hallucinated-hallucinated, and hallucinated-real distances
    real_real_distances = np.array([data['distance'] for data in dataset["right_right"]])
    hallucinated_hallucinated_distances = np.array([data['distance'] for data in dataset["halu_halu"]])
    hallucinated_real_distances = np.array([data['distance'] for data in dataset["right_halu"]])
    total_distances = np.concatenate([real_real_distances, hallucinated_hallucinated_distances, hallucinated_real_distances])
    
    # Set bin width and calculate bin edges
    bin_width = total_distances.std() / 5
    # bin_width = total_distances.mean() / 5

    # bin_width = 0.005
    print(f"Bin width: {bin_width}")
    min_dist, max_dist = min(total_distances), max(total_distances)
    bins = np.arange(min_dist, max_dist + bin_width, bin_width)
    
    # Compute histograms for the three categories
    real_real_hist, _ = np.histogram(real_real_distances, bins=bins)
    hallucinated_hallucinated_hist, _ = np.histogram(hallucinated_hallucinated_distances, bins=bins)
    hallucinated_real_hist, _ = np.histogram(hallucinated_real_distances, bins=bins)
    
    # Calculate total counts in each bin
    total_counts = real_real_hist + hallucinated_hallucinated_hist + hallucinated_real_hist
    
    # Avoid division by zero for bins with no counts
    real_real_prob = np.where(total_counts == 0, 0, real_real_hist / total_counts)
    hallucinated_hallucinated_prob = np.where(total_counts == 0, 0, hallucinated_hallucinated_hist / total_counts)
    hallucinated_real_prob = np.where(total_counts == 0, 0, hallucinated_real_hist / total_counts)
    
    # Calculate the percentage for each category
    real_real_percent = real_real_prob * 100
    hallucinated_hallucinated_percent = hallucinated_hallucinated_prob * 100
    hallucinated_real_percent = hallucinated_real_prob * 100
    
    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the histograms with overlap and KDEs
    plt.hist(real_real_distances, bins=bins, alpha=0.7, label="Real-Real Text", color='blue', weights=np.ones_like(real_real_distances) * 100. / len(real_real_distances))
    plt.hist(hallucinated_hallucinated_distances, bins=bins, alpha=0.7, label="Hallucinated-Hallucinated Text", color='red', weights=np.ones_like(hallucinated_hallucinated_distances) * 100. / len(hallucinated_hallucinated_distances))
    # plt.hist(hallucinated_real_distances, bins=bins, alpha=0.7, label="Hallucinated-Real Text", color='green', weights=np.ones_like(hallucinated_real_distances) * 100. / len(hallucinated_real_distances))

    # Optional: Add KDE plot for smoother visualization
    # sns.kdeplot(real_real_distances, color='blue', fill=True, alpha=0.3, label='KDE Real-Real Text')
    # sns.kdeplot(hallucinated_hallucinated_distances, color='red', fill=True, alpha=0.3, label='KDE Hallucinated-Hallucinated Text')
    
    plt.title("Overlapping Probability Distribution of Hallucinated vs. Real Text Pairs")
    plt.xlabel(f"Distance (Bin width = {bin_width})")
    plt.ylabel("Percentage of Text")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True)
    plt.show()

    overlap = np.sum(np.minimum(real_real_hist, hallucinated_hallucinated_hist)) / np.sum(real_real_hist)
    print(f'Overlap between Real-Real and Hallucinated-Hallucinated: {overlap:.8f}')

def plot_rouge_score_distribution_histogram(dataset, rouge_metric='rouge2'):
    """
    Plots overlapping probability distribution histograms for specified ROUGE scores
    between real-real, hallucinated-hallucinated, and real-hallucinated text pairs.
    
    Parameters:
    - dataset (dict): Dictionary containing 'right_right', 'halu_halu', and 'right_halu' keys with lists of ROUGE scores.
    - rouge_metric (str): The ROUGE metric to plot (e.g., 'rouge1', 'rouge2', 'rougeL').
    
    Returns:
    - None: Displays the histogram plot.
    """
    # Extract ROUGE scores for each category
    real_real_rouge = np.array([data['rouge_score'][rouge_metric][2] for data in dataset.get("right_right", [])])
    halu_halu_rouge = np.array([data['rouge_score'][rouge_metric][2] for data in dataset.get("halu_halu", [])])
    real_halu_rouge = np.array([data['rouge_score'][rouge_metric][2] for data in dataset.get("right_halu", [])])
    
    # Combine all ROUGE scores to determine bin settings
    all_rouge_scores = np.concatenate([real_real_rouge, halu_halu_rouge, real_halu_rouge])
    
    if len(all_rouge_scores) == 0:
        print("No ROUGE scores available to plot.")
        return
    
    # Define the number of bins (e.g., 30 bins)
    num_bins = 30
    bins = np.linspace(0, 1, num_bins + 1)  # ROUGE scores range from 0 to 1
    
    # Compute histograms for each category
    real_real_hist, _ = np.histogram(real_real_rouge, bins=bins)
    halu_halu_hist, _ = np.histogram(halu_halu_rouge, bins=bins)
    real_halu_hist, _ = np.histogram(real_halu_rouge, bins=bins)
    
    # Calculate total counts in each bin for normalization
    total_counts = real_real_hist + halu_halu_hist + real_halu_hist
    
    # Avoid division by zero
    real_real_prob = np.where(total_counts == 0, 0, real_real_hist / total_counts)
    halu_halu_prob = np.where(total_counts == 0, 0, halu_halu_hist / total_counts)
    real_halu_prob = np.where(total_counts == 0, 0, real_halu_hist / total_counts)
    
    # Calculate percentages
    real_real_percent = real_real_prob * 100
    halu_halu_percent = halu_halu_prob * 100
    real_halu_percent = real_halu_prob * 100
    
    # Plotting
    plt.figure(figsize=(12, 7))

    # Plot histograms for each category
    plt.hist(real_real_rouge, bins=bins, alpha=0.6, label="Real-Real Text", color='blue', weights=np.ones_like(real_real_rouge) * 100. / len(real_real_rouge))
    plt.hist(halu_halu_rouge, bins=bins, alpha=0.6, label="Hallucinated-Hallucinated Text", color='red', weights=np.ones_like(halu_halu_rouge) * 100. / len(halu_halu_rouge))
    # plt.hist(real_halu_rouge, bins=bins, alpha=0.6, label="Real-Hallucinated Text", color='green', weights=np.ones_like(real_halu_rouge) * 100. / len(real_halu_rouge))

    plt.title(f"Probability Distribution of ROUGE-{rouge_metric.upper()} Scores")
    plt.xlabel(f"ROUGE-{rouge_metric.upper()} Score")
    plt.ylabel("Percentage of Text Pairs (%)")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Optional: Calculate and print overlap between Real-Real and Hallucinated-Hallucinated
    overlap = np.sum(np.minimum(real_real_hist, halu_halu_hist)) / np.sum(real_real_hist)
    print(f'Overlap between Real-Real and Hallucinated-Hallucinated ROUGE-{rouge_metric.upper()}: {overlap:.4f}')



def heatmap(procrustes_distances):
    # Initialize an empty list to collect all records
    records = []

    # Iterate over each pair type and its list of distance records
    for pair_type, distance_list in procrustes_distances.items():
        for record in distance_list:
            # Add a new key for the pair type
            record_copy = record.copy()
            record_copy['pair_type'] = pair_type
            records.append(record_copy)

    # Create a DataFrame from the records
    df_distances = pd.DataFrame(records)

    # Reorder columns for clarity
    df_distances = df_distances[['question', 'pair_type', 'pair_indices', 'distance']]

    # Step 1: Aggregate Mean Distances
    mean_distances = df_distances.groupby(['question', 'pair_type'])['distance'].mean().reset_index()

    # Step 2: Pivot the DataFrame
    pivot_df = mean_distances.pivot(index='question', columns='pair_type', values='distance').fillna(0)

    # Step 3: Create the Heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)

    # Step 4: Customize the Heatmap
    plt.title('Mean Procrustes Distances per Question and Pair Type')
    plt.xlabel('Pair Type')
    plt.ylabel('Question')

    # Step 5: Display the Heatmap
    plt.show()

def plot_scatter(x, y, x_title, y_title, title):
    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, alpha=0.7, color='blue')

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.show()

def print_statistics(procrustes_distances):
    # Initialize an empty list to collect all records
    records = []

    # Iterate over each pair type and its list of distance records
    for pair_type, distance_list in procrustes_distances.items():
        for record in distance_list:
            # Add a new key for the pair type
            record_copy = record.copy()
            record_copy['pair_type'] = pair_type
            records.append(record_copy)

    # Create a DataFrame from the records
    df_distances = pd.DataFrame(records)

    # Reorder columns for clarity
    df_distances = df_distances[['question', 'pair_type', 'pair_indices', 'distance']]

    # Group by 'question' and 'pair_type' to compute mean distance
    mean_distances = df_distances.groupby(['question', 'pair_type'])['distance'].mean().reset_index()

    # Pivot the DataFrame
    pivot_df = mean_distances.pivot(index='question', columns='pair_type', values='distance').reset_index()

    # Calculate the difference per question
    pivot_df['difference'] = pivot_df['halu_halu'] - pivot_df['right_right']

    print("\nMean Distance Differences Per Question:")
    print(pivot_df[['question', 'difference']])

    # Aggregate the differences across all questions
    aggregate_stats = pivot_df['difference'].agg(['mean', 'median', 'std'])

    print("\nAggregate Statistics of Differences:")
    print(aggregate_stats)


if __name__ == '__main__':
    '''
    texts = [
        # "I love machine learning",
        # "Machine learning is fun",
        "I love coding",
        # "Coding is also fun"
    ]

    # Create a sample vocabulary
    vocab = {
        'i': 0,
        'love': 1,
        'machine': 2,
        'learning': 3,
        'is': 4,
        'fun': 5,
        'coding': 6,
        'also': 7
    }
    cooccurrence_matrix = create_cooccurrence_matrix(texts, vocab, ngram_range=(2, 2))
    # Optionally: print the matrix with vocab words for better readability
    print("\nCo-occurrence Matrix with Vocabulary Mapping:")
    vocab_list = list(vocab.keys())
    print(f"{'':8}", end='')
    for word in vocab_list:
        print(f"{word:8}", end='')
    print()

    for i, word1 in enumerate(vocab_list):
        print(f"{word1:8}", end='')
        for j in range(len(vocab)):
            print(f"{cooccurrence_matrix[i, j]:8}", end='')
        print()
    exit()
    '''

    # unrandomized dataset 
    dataset = create_dataset('data/TruthfulQA.csv', first=False)

    # randomized dataset
    randomized_dataset = randomize_dataset(dataset)

    # only use first answer 

    vocab = create_global_vocab(randomized_dataset)

    all_texts = randomized_dataset["answer"]
    real_texts = dataset["right_answer"]
    halu_texts = dataset["halu_answer"]

    # Convert to DataFrame
    df = pd.DataFrame(pad_lists(randomized_dataset))

    # Initialize new columns for latent vectors
    df['right_answer_latent'] = None
    df['halu_answer_latent'] = None

    # Parameters 
    ngram = 2
    ngram_range = (2, 2)  # Bigrams
    n_svd_components = 15 # Number of latent concepts

    # Group the dataset by 'question'
    # Initialize new dictionaries to store latent vectors and distances
    latent_dataset = {
        "question": [],
        "right_answer": [],
        "halu_answer": [],
        "right_answer_text": [],
        "halu_answer_text": []
    }

    procrustes_distances = {
        "right_right": [],
        "halu_halu": [],
        "right_halu": []
    }

    # Group the dataset by 'question' to ensure unified feature space per question
    grouped = df.groupby('question')
    latent_vectors_real = []
    latent_vectors_halu = []

    i = 0 
    for question, group in grouped:
        # if i < 7:  # Check if i is less than 1
        #     i += 1  # Increment i
        #     continue  # Skip this iteration

        print(f"Processing question: {question}")
        
        # Extract real and hallucinated answers
        real_answers = group[group['label'] == 1]['answer'].dropna().tolist()
        halu_answers = group[group['label'] == 0]['answer'].dropna().tolist()
        
        # Create a unified vocabulary for the current question
        local_vocab = create_local_vocab(real_answers, halu_answers, 1)

        # Create a unified n-gram matrix
        # X_real, X_halu, local_vocab = create_unified_ngram_matrix(real_answers, halu_answers, ngram_range)
        # X_real, X_halu, local_vocab = create_unified_ngram_matrix(real_answers, halu_answers, ngram_range)

        X_real = create_cooccurrence_matrix(real_answers, local_vocab, ngram_range)
        X_halu = create_cooccurrence_matrix(halu_answers, local_vocab, ngram_range)

        latent_vectors_real = []
        latent_vectors_halu = []


        # Process each batch in X_real
        for i in range(X_real.shape[0]):
            batch_real = X_real[i]

            if batch_real.shape[0] > 0 and batch_real.shape[1] > 1:
                try:
                    # Perform full SVD
                    U, S, VT = np.linalg.svd(batch_real, full_matrices=True)
                    # # Visualize the Sigma matrix using a heatmap
                    # plt.figure(figsize=(6, 4))
                    # plt.imshow(np.diag(S), cmap='viridis', aspect='auto')
                    # plt.colorbar(label='Singular Value Magnitude')
                    # plt.title('Sigma Matrix Visualization')
                    # plt.xlabel('Columns')
                    # plt.ylabel('Rows')
                    # plt.show()
                    # np.set_printoptions(threshold=np.inf)
                    # print(S)
                    # print(batch_real)
                    # break

                    # Take only the top n_svd_components
                    latent_vectors = np.dot(U[:, :n_svd_components], np.diag(S[:n_svd_components])) # SHAPE IS (2,2) UNLESS WE TRANSPOSE
                    # latent_vectors = U[:, :n_svd_components]  # SHAPE IS (2,2)
                    # latent_vectors = VT[:n_svd_components, :].T
                    latent_vectors_real.append(latent_vectors.tolist())
                except ValueError as e:
                    print(f"Error performing SVD on real answers batch {i}: {e}")
            else:
                print(f"Skipping real answers batch {i} due to insufficient data for SVD.")

        # Process each batch in X_halu
        for i in range(X_halu.shape[0]):
            batch_halu = X_halu[i]
            
            if batch_halu.shape[0] > 0 and batch_halu.shape[1] > 1:
                try:
                    # Perform full SVD
                    U, S, VT = np.linalg.svd(batch_halu, full_matrices=True)
                    # # Visualize the Sigma matrix using a heatmap
                    # plt.figure(figsize=(6, 4))
                    # plt.imshow(np.diag(S), cmap='viridis', aspect='auto')
                    # plt.colorbar(label='Singular Value Magnitude')
                    # plt.title('Sigma Matrix Visualization')
                    # plt.xlabel('Columns')
                    # plt.ylabel('Rows')
                    # plt.show()
                    # break

                    # Take only the top n_svd_components
                    latent_vectors = np.dot(U[:, :n_svd_components], np.diag(S[:n_svd_components]))
                    # latent_vectors = U[:, :n_svd_components]
                    # latent_vectors = VT[:n_svd_components, :].T
                    latent_vectors_halu.append(latent_vectors.tolist())
                except ValueError as e:
                    print(f"Error performing SVD on hallucinated answers batch {i}: {e}")
            else:
                print(f"Skipping hallucinated answers batch {i} due to insufficient data for SVD.")

        latent_dataset['question'].append(question)
        latent_dataset['right_answer'].append(latent_vectors_real)
        latent_dataset['halu_answer'].append(latent_vectors_halu)
        latent_dataset['right_answer_text'].append(real_answers)
        latent_dataset['halu_answer_text'].append(halu_answers)
        # break


    # Update the DataFrame with the latent vectors    
    # Display the updated DataFrame
    # for index, row in df.iterrows():
    #     print(f"\nRow {index}:")
    #     print(f"Question: {row['question']}")
    #     print(f"Right Answer: {row['right_answer']}")
    #     print(f"Right Answer Latent: {row['right_answer_latent']}")
    #     print(f"Halu Answer: {row['halu_answer']}")
    #     print(f"Halu Answer Latent: {row['halu_answer_latent']}")


    latent_df = pd.DataFrame(latent_dataset)
    grouped = latent_df.groupby('question')

    # # Display the latent dataset
    # print("\nLatent Dataset:")
    # pp.pprint(latent_dataset)

    print("Calculating Procrustes Distances...")

    for question, group in grouped:
        current_right_latents = np.array(group['right_answer'])[0]
        current_halu_latents = np.array(group['halu_answer'])[0]
        current_right_texts = np.array(group['right_answer_text'])[0]
        current_halu_texts = np.array(group['halu_answer_text'])[0]
        # Compute right_answer ↔ right_answer distances
        for i, j in itertools.combinations(range(len(current_right_latents)), 2):
            # calcualte rouge score 
            rouge_score = scorer.score(current_right_texts[i], current_right_texts[j])

            ra1 = np.array(current_right_latents[i])
            ra2 = np.array(current_right_latents[j])
            if np.array_equal(ra1, ra2):
                continue  # Skip identical vectors
            try:
                row, col = ra1.shape

                _, _, disparity = procrustes(ra1.reshape(col, row), ra2.reshape(col, row))
                procrustes_distances["right_right"].append({
                    "question": question,
                    "pair_indices": (i, j),
                    "distance": disparity,
                    "rouge_score": rouge_score
                })
            except ValueError as e:
                print(f"Skipping pair {(i, j)} for question '{question}' due to error: {e}")
                continue  # Skip this pair

        # Compute halu_answer ↔ halu_answer distances
        for i, j in itertools.combinations(range(len(current_halu_latents)), 2):
            # calcualte rouge score 
            rouge_score = scorer.score(current_halu_texts[i], current_halu_texts[j])

            ha1 = np.array(current_halu_latents[i])
            ha2 = np.array(current_halu_latents[j])
            if np.array_equal(ha1, ha2):
                continue  # Skip identical vectors
            try:
                row, col = ha1.shape
                _, _, disparity = procrustes(ha1.reshape(col, row), ha2.reshape(col, row))
                procrustes_distances["halu_halu"].append({
                    "question": question,
                    "pair_indices": (i, j),
                    "distance": disparity,
                    "rouge_score": rouge_score
                })
            except ValueError as e:
                print(f"Skipping pair {(i, j)} for question '{question}' due to error: {e}")
                continue  # Skip this pair

        # Compute right_answer ↔ halu_answer distances
        for i in range(len(current_right_latents)):
            for j in range(len(current_halu_latents)):
                # calcualte rouge score 
                rouge_score = scorer.score(current_right_texts[i], current_halu_texts[j])

                ra = np.array(current_right_latents[i])
                ha = np.array(current_halu_latents[j])
                if np.array_equal(ra, ha):
                    continue  # Skip identical vectors
                try:
                    row1, col1 = ha.shape
                    row2, col2 = ra.shape
                    _, _, disparity = procrustes(ra.reshape(col2, row2), ha.reshape(col1, row1))
                    procrustes_distances["right_halu"].append({
                        "question": question,
                        "pair_indices": (i, j),
                        "distance": disparity,
                        "rouge_score": rouge_score
                    })
                except ValueError as e:
                    print(f"Skipping pair ({i}, {j}) for question '{question}' due to error: {e}")
                    continue  # Skip this pair

    # Display the distance distances
    print("\nProcrustes Distances:")
    pp.pprint(procrustes_distances)

    # Optionally, save the distances to a file
    with open('procrustes_distances.json', 'w') as f:
        json.dump(procrustes_distances, f, indent=2)

    # Reload the distances from the file
    with open('procrustes_distances.json', 'r') as f:
        procrustes_distances = json.load(f)

    plot_probability_distribution_histogram(procrustes_distances)
    plot_rouge_score_distribution_histogram(procrustes_distances)


    # Filter for 'right_hallu' pair_type and identical pair_indices
    filtered_data = [
        entry for entry in procrustes_distances["right_halu"]
        if entry['pair_indices'][0] == entry['pair_indices'][1]
    ]
    
    # Extract Procrustes distances, ROUGE-2 F1 scores, and labels
    procrustes_distances_list = []
    rouge2_fmeasure = []
    labels = []
    
    for entry in filtered_data:
        distance = entry.get('distance')
        rouge2 = entry.get('rouge_score', {}).get('rouge2', [])
        
        # Ensure that rouge2 has at least 3 elements (precision, recall, fmeasure)
        if distance is not None and isinstance(rouge2, list) and len(rouge2) >= 3:
            procrustes_distances_list.append(distance)
            rouge2_fmeasure.append(rouge2[2])  # ROUGE-2 F1 score
            labels.append(f"{entry['pair_indices'][0]}-{entry['pair_indices'][1]}")
        else:
            print(f"Skipping entry due to missing data: {entry}")
    
    # Check if all lists have the same length
    if not (len(procrustes_distances_list) == len(rouge2_fmeasure) == len(labels)):
        print("Error: Procrustes distances, ROUGE scores, and labels lists have different lengths.")
    plot_scatter(
        x=procrustes_distances_list,
        y=rouge2_fmeasure,
        x_title='Procrustes Distance',
        y_title='ROUGE-2 F1 Score',
        title='Procrustes Distance vs ROUGE-2 F1 Score for Right-Hallu Pairs'
    )

    # heatmap(procrustes_distances)
    print_statistics(procrustes_distances)



