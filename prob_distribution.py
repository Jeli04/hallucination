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
from matplotlib.ticker import PercentFormatter
import pickle

# Ensure you have the required data files
nltk.download('punkt')


def load_data(file_path, type='json'):
    if type == 'json':
        with open(file_path, 'r', encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif type == "csv":
        data = pd.read_csv(file_path)
    return data


def create_vocab(data, keys=['right_answer']):
    vocab = {}

    if "Correct Answers" in keys or "Incorrect Answers" in keys:
        for i, entry in data.iterrows():
            for key in keys:
                answer_list = entry[key].split(';')
                words = nltk.word_tokenize(answer_list[0])  # only gets the first
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)  # Assign a unique ID
    else:
        for entry in data:
            for key in keys:
                words = nltk.word_tokenize(entry.get(key))
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)  # Assign a unique ID
    return vocab

def create_vocab2(text1, text2):
    vocab = {}
    word_list = nltk.word_tokenize(text1) + nltk.word_tokenize(text2)
    for word in word_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def generate_word_ngrams(text, n):
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    ngrams_list = list(ngrams(words, n))  # Generate n-grams
    return [list(tup) for tup in ngrams_list]


def prepare_svd_input(grams, vocab, ngram):
    input = []
    for i in range(ngram):
        hash_map = [0] * len(vocab)
        for gram in grams:
            hash_map[vocab[gram[i]]] += 1
        input.append(hash_map)
    return torch.tensor(input, dtype=torch.float)  # A tensor of shape (len(vocab), ngram)


def compute_procrustes_distances(tensor1, tensor2):
    try:
        if tensor1.size(0) == 1 and tensor2.size(0) == 1:
            return 0.0
        _, _, disparity = procrustes(tensor1.cpu().numpy(), tensor2.cpu().numpy())
        return disparity
    except ValueError as e:
        if str(e) == "Input matrices must contain >1 unique points":
            # print("Procrustes analysis error: Input matrices must contain more than 1 unique point.")
            # print(tensor1, tensor2)
            return 0.0
        else:
            raise  # Re-raise the error if it's not the one we're handling


def compute_cka(tensor1, tensor2):
    # Compute the Gram matrices for both tensors
    K_X = tensor1 @ tensor1.T
    K_Y = tensor2 @ tensor2.T

    # Center the Gram matrices
    H = torch.eye(K_X.size(0), device=tensor1.device) - (1.0 / K_X.size(0)) * torch.ones_like(K_X)
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Compute the CKA score
    cka_score = (K_X_centered * K_Y_centered).sum() / (
        torch.sqrt((K_X_centered ** 2).sum()) * torch.sqrt((K_Y_centered ** 2).sum()))

    return cka_score.item()


def calculate_rouge_n(actual_text: str, hallucinated_text: str, n: int) -> Tuple[float, float, float]:
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    scores = scorer.score(actual_text, hallucinated_text)
    precision = scores[f'rouge{n}'].precision
    recall = scores[f'rouge{n}'].recall
    f1_score = scores[f'rouge{n}'].fmeasure
    return precision, recall, f1_score

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj



def plot_probability_distribution_histogram(distances, bin_width=0.1):
    # Separate real and hallucinated text distances
    real_distances = np.array([distances[i] for i in range(0, len(distances), 2)])
    hallucinated_distances = np.array([distances[i] for i in range(1, len(distances), 2)])
    
    # Determine bin edges based on the minimum and maximum distances, adjusted by bin width
    min_dist, max_dist = min(distances), max(distances)
    bins = np.arange(min_dist, max_dist + bin_width, bin_width)
    
    # Compute histograms for real and hallucinated distances
    real_hist, _ = np.histogram(real_distances, bins=bins)
    hallucinated_hist, _ = np.histogram(hallucinated_distances, bins=bins)
    
    # Calculate total counts in each bin
    total_counts = real_hist + hallucinated_hist
    
    # Avoid division by zero for bins with no counts
    real_prob = np.where(total_counts == 0, 0, real_hist / total_counts)
    hallucinated_prob = np.where(total_counts == 0, 0, hallucinated_hist / total_counts)
    
    # Calculate the percentage for real and hallucinated
    real_percent = real_prob * 100
    hallucinated_percent = hallucinated_prob * 100
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the real and hallucinated histograms with overlap
    plt.hist(real_distances, bins=bins, alpha=0.7, label="Real-Real Text", color='blue', weights=np.ones_like(real_distances) * 100. / len(real_distances))
    plt.hist(hallucinated_distances, bins=bins, alpha=0.7, label="Hallucinated-Hallucinated Text", color='red', weights=np.ones_like(hallucinated_distances) * 100. / len(hallucinated_distances))
    
    plt.title("Overlapping Probability Distribution of Hallucinated vs. Real Text Pairs")
    plt.xlabel(f"Distance (Bin width = {bin_width})")
    plt.ylabel("Percentage of Text")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True)
    plt.show()

def plot_probability_distribution_histogram2(distances, bin_width=0.1):
    # Separate real-real, hallucinated-hallucinated, and hallucinated-real distances
    real_real_distances = np.array([distances[i] for i in range(0, len(distances), 5)])
    hallucinated_hallucinated_distances = np.array([distances[i] for i in range(1, len(distances), 5)])
    hallucinated_real_distances = np.array([distances[i+j] for i in range(2, len(distances), 5) for j in range(3)])

    # Determine bin edges based on the minimum and maximum distances, adjusted by bin width
    min_dist, max_dist = min(distances), max(distances)
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
    
    # Plot the histograms with overlap
    plt.hist(real_real_distances, bins=bins, alpha=0.7, label="Real-Real Text", color='blue', weights=np.ones_like(real_real_distances) * 100. / len(real_real_distances))
    plt.hist(hallucinated_hallucinated_distances, bins=bins, alpha=0.7, label="Hallucinated-Hallucinated Text", color='red', weights=np.ones_like(hallucinated_hallucinated_distances) * 100. / len(hallucinated_hallucinated_distances))
    plt.hist(hallucinated_real_distances, bins=bins, alpha=0.7, label="Hallucinated-Real Text", color='green', weights=np.ones_like(hallucinated_real_distances) * 100. / len(hallucinated_real_distances))
    
    plt.title("Overlapping Probability Distribution of Hallucinated vs. Real Text Pairs")
    plt.xlabel(f"Distance (Bin width = {bin_width})")
    plt.ylabel("Percentage of Text")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    data = load_data('data/qa_data.json')

    file_path = 'data/TruthfulQA.csv'
    data = pd.read_csv(file_path)

    svd_setting = "col"
    ngram = 2
    k = 2 # low rank approximation
    cosine_similarities = []
    procrustes_distances = []
    cka_scores = []
    rouge_scores = []
    modified_data = []

    # Create the tokenized vocabulary
    # right_vocab = create_vocab(data, keys=['right_answer'])
    # halu_vocab = create_vocab(data, keys=['hallucinated_answer'])
    right_vocab = create_vocab(data, keys=['Correct Answers'])
    halu_vocab = create_vocab(data, keys=['Incorrect Answers'])

    # Union the vocabularies
    vocab = {**right_vocab, **halu_vocab}

    # very inefficient ill fix later lol
    for i, (index, entry) in tqdm(enumerate(data.iterrows()), total=len(data), desc="Processing Entries"):
        correct_answers= entry['Correct Answers'].split(';')[0]
        incorrect_answers = entry['Incorrect Answers'].split(';')[0]

        question_grams = generate_word_ngrams(entry['Question'], ngram)
        right_grams = generate_word_ngrams(correct_answers, ngram)
        hallucinated_grams = generate_word_ngrams(incorrect_answers, ngram)

        # Prepare the input for SVD
        right_input = prepare_svd_input(right_grams, vocab, ngram).to(device)
        halu_input = prepare_svd_input(hallucinated_grams, vocab, ngram).to(device)

        # Perform SVD and low rank approximation
        U, S, V = torch.linalg.svd(right_input)
        U = U[:, :k]
        S = S[:k]
        V = V[:, :k]
        if svd_setting == "row":
            Xo = torch.mm(U, torch.diag(S))
        elif svd_setting == "col":
            Xo = V
            
        U, S, V = torch.linalg.svd(halu_input)
        U = U[:, :k]
        S = S[:k]
        V = V[:, :k]
        if svd_setting == "row":
            Xg = torch.mm(U, torch.diag(S))
        elif svd_setting == "col":
            Xg = V
                

        for j, entry2 in data.iloc[i+1:].iterrows(): 
            correct_answers2= entry2['Correct Answers'].split(';')[0]
            incorrect_answers2 = entry2['Incorrect Answers'].split(';')[0]

            # union the input vocabularies
            # input_vocab = create_vocab2(correct_answers_list, incorrect_answers_list)

            question_grams2 = generate_word_ngrams(entry['Question'], ngram)
            right_grams2 = generate_word_ngrams(correct_answers2, ngram)
            hallucinated_grams2 = generate_word_ngrams(incorrect_answers2, ngram)

            # Prepare the input for SVD
            right_input2 = prepare_svd_input(right_grams2, vocab, ngram).to(device)
            halu_input2 = prepare_svd_input(hallucinated_grams2, vocab, ngram).to(device)

            # Perform SVD and low rank approximation
            U, S, V = torch.linalg.svd(right_input, full_matrices=True)
            U = U[:, :k]
            S = S[:k]
            V = V[:, :k]
            if svd_setting == "row":
                Xo2 = torch.mm(U, torch.diag(S))
            elif svd_setting == "col":
                Xo2 = V
                
            U, S, V = torch.linalg.svd(halu_input, full_matrices=True)
            U = U[:, :k]
            S = S[:k]
            V = V[:, :k]
            if svd_setting == "row":
                Xg2 = torch.mm(U, torch.diag(S))
            elif svd_setting == "col":
                Xg2 = V
                    
            # EVEN IS REAL ODD IS HALUCINATED
            # Compute the Procrustes analysis
            disparity = compute_procrustes_distances(Xo, Xo2)
            procrustes_distances.append(disparity)

            disparity = compute_procrustes_distances(Xg, Xg2)
            procrustes_distances.append(disparity)

            # compute every real to halu pair also
            disparity = compute_procrustes_distances(Xg, Xo2)
            procrustes_distances.append(disparity)

            disparity = compute_procrustes_distances(Xg2, Xo)
            procrustes_distances.append(disparity)

            disparity = compute_procrustes_distances(Xg, Xo)
            procrustes_distances.append(disparity)


    # Save the list to a file
    with open('distances.pkl', 'wb') as file:
        pickle.dump(procrustes_distances, file)
    

    # Later, to load the list back from the file
    with open('distances.pkl', 'rb') as file:
        procrustes_distances = pickle.load(file)

    # Plot the probability distribution
    plot_probability_distribution_histogram2(procrustes_distances, bin_width=0.01)
