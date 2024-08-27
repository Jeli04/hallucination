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
    for entry in data:
        for key in keys:
            words = nltk.word_tokenize(entry.get(key))  # Tokenize the text into words
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)  # Assign a unique ID
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
    return torch.tensor(input, dtype=torch.float).T  # A tensor of shape (len(vocab), ngram)


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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    data = load_data('data/qa_data.json')

    svd_setting = "col"
    ngram = 2
    cosine_similarities = []
    procrustes_distances = []
    cka_scores = []
    rouge_scores = []
    modified_data = []

    # Create the tokenized vocabulary
    right_vocab = create_vocab(data, keys=['right_answer'])
    halu_vocab = create_vocab(data, keys=['hallucinated_answer'])

    # Union the vocabularies
    vocab = {**right_vocab, **halu_vocab}

    # Intersect the vocabularies
    # vocab = create_vocab(data, keys=['right_answer', 'hallucinated_answer'])

    # Add a progress bar to the loop
    for i, entry in tqdm(enumerate(data), total=len(data), desc="Processing Entries"):
        question_grams = generate_word_ngrams(entry['question'], ngram)
        right_grams = generate_word_ngrams(entry['right_answer'], ngram)
        hallucinated_grams = generate_word_ngrams(entry['hallucinated_answer'], ngram)

        # Prepare the input for SVD
        right_input = prepare_svd_input(right_grams, vocab, ngram).to(device)
        halu_input = prepare_svd_input(hallucinated_grams, vocab, ngram).to(device)

        # Perform SVD
        U, S, V = torch.svd(right_input)
        Xo = U if svd_setting == "row" else torch.mm(U, torch.diag(S))

        U, S, V = torch.svd(halu_input)
        Xg = U if svd_setting == "row" else torch.mm(U, torch.diag(S))

        # Compute the cosine similarity between the two matrices
        cosine_similarity = torch.nn.functional.cosine_similarity(Xo, Xg, dim=1)
        cosine_similarities.append(torch.mean(cosine_similarity).item())

        # Compute the Procrustes analysis
        disparity = compute_procrustes_distances(Xo, Xg)
        procrustes_distances.append(disparity)

        # Compute the CKA similarity
        # cka_score = compute_cka(Xo, Xg)
        # cka_scores.append(cka_score)

        # Calculate the Rouge-N score between the right and hallucinated answers
        rouge_scores.append(calculate_rouge_n(entry['right_answer'], entry['hallucinated_answer'], ngram))

        # Store modified data in the new variable
        modified_entry = {
            'cosine': torch.mean(cosine_similarity).item(),
            'procrustes': disparity,
            # 'cka': cka_score,
            'right_grams': len(right_grams),
            'hallucinated_grams': len(hallucinated_grams),
            'question_grams': len(question_grams),
            'rouge': rouge_scores[i]
        }
        modified_data.append(modified_entry)

    print("-" * 50)
    print("Average Cosine Similarity:", np.mean(cosine_similarities))
    print("Min Cosine Similarity:", np.min(cosine_similarities))
    print("Max Cosine Similarity:", np.max(cosine_similarities))
    print("Standard Deviation of Cosine Similarity:", np.std(cosine_similarities))
    print("-" * 50)
    print("Average Procrustes Distance:", np.mean(procrustes_distances))
    print("Min Procrustes Distance:", np.min(procrustes_distances))
    print("Max Procrustes Distance:", np.max(procrustes_distances))
    print("Standard Deviation of Procrustes Distance:", np.std(procrustes_distances))
    # print("-" * 50)
    # print("Average CKA Score:", np.mean(cka_scores))
    # print("Min CKA Score:", np.min(cka_scores))
    # print("Max CKA Score:", np.max(cka_scores))
    # print("Standard Deviation of CKA Score:", np.std(cka_scores))
    print("-" * 50)
    print(f"Average {ngram}-Rouge Score:", np.mean(rouge_scores))
    print("Min Rouge Score:", np.min(rouge_scores))
    print("Max Rouge Score:", np.max(rouge_scores))
    print("Standard Deviation of Rouge Score:", np.std(rouge_scores))
    print("-" * 50)

    # Write the modified data to a new JSON file
    with open('modified_data.json', 'w') as f:
        json.dump(modified_data, f, indent=4, default=convert_to_serializable)

