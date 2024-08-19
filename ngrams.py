import torch
import json
import nltk
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial import procrustes
from rouge_score import rouge_scorer
from typing import Tuple

# Ensure you have the required data files
nltk.download('punkt')

file_path = "data/qa_data.json"
with open(file_path, 'r', encoding="utf-8") as f:
    data = []
    for line in f:
        data.append(json.loads(line))


# Function to tokenize text and update vocabulary
def tokenize_and_update_vocab(text, vocab):
    tokenized = []
    for gram in text:
        pair = []
        for word in gram:
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Assign a unique ID starting from 1
            pair.append(vocab[word])
        tokenized.append(pair)
    return tokenized

def generate_word_ngrams(text, n):
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    
    if len(words) < n:  # Check if the text is too short to generate n-grams
        words = words + (['<PAD>']  * (n - len(words))) # Pad the text with "<PAD>" tokens

    ngrams_list = list(ngrams(words, n))  # Generate n-grams

    ngrams_list = [list(tup) for tup in ngrams_list]

    tokenized_ngrams = tokenize_and_update_vocab(ngrams_list, vocab)  # Tokenize the n-grams

    return tokenized_ngrams

def pad_grams(grams, n):
    # Determine the maximum length of the inner lists
    max_length = max(len(inner_list) for inner_list in grams)

    # Pad each inner list to the maximum length
    grams = [inner_list + [[0]*n] * (max_length - len(inner_list)) for inner_list in grams]

    return torch.tensor(grams, dtype=torch.float)

def pad_to_same_size(tensor1, tensor2):
    # Get the shapes of the tensors
    rows1, cols1 = tensor1.shape
    rows2, cols2 = tensor2.shape
    
    # Determine the new size
    new_rows = max(rows1, rows2)
    new_cols = max(cols1, cols2)

    if new_cols == 1:
        new_cols = ngram
    if new_rows == 1:
        new_rows = ngram   
    
    # Create new tensors with the new size, filled with zeros
    padded_tensor1 = torch.zeros((new_rows, new_cols), dtype=tensor1.dtype)
    padded_tensor2 = torch.zeros((new_rows, new_cols), dtype=tensor2.dtype)
    
    # Copy the values of the original tensors into the new tensors
    padded_tensor1[:rows1, :cols1] = tensor1
    padded_tensor2[:rows2, :cols2] = tensor2
    
    return padded_tensor1, padded_tensor2

# Function to compute Procrustes distance for each pair of inner tensors
def compute_procrustes_distances(tensor1, tensor2):
    try:
        if tensor1.size(0) == 1 and tensor2.size(0) == 1:
            return 0.0
        _, _, disparity = procrustes(tensor1, tensor2)
        return disparity
    except ValueError as e:
        if str(e) == "Input matrices must contain >1 unique points":
            print("Procrustes analysis error: Input matrices must contain more than 1 unique point.")
            print(tensor1, tensor2)
            exit()
        else:
            raise  # Re-raise the error if it's not the one we're handling

# Function to compute CKA similarity between two tensors
def compute_cka(tensor1, tensor2):
    # Compute the Gram matrices for both tensors
    K_X = tensor1 @ tensor1.T
    K_Y = tensor2 @ tensor2.T
    
    # Center the Gram matrices
    H = torch.eye(K_X.size(0)) - (1.0 / K_X.size(0)) * torch.ones_like(K_X)
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H
    
    # Compute the CKA score
    cka_score = (K_X_centered * K_Y_centered).sum() / (
        torch.sqrt((K_X_centered ** 2).sum()) * torch.sqrt((K_Y_centered ** 2).sum()))
    
    return cka_score.item()

def calculate_rouge_n(actual_text: str, hallucinated_text: str, n: int) -> Tuple[float, float, float]:
    """
    Calculate the ROUGE-N score between actual and hallucinated text.
    
    Parameters:
    actual_text (str): The reference text.
    hallucinated_text (str): The generated text.
    n (int): The n-gram length.

    Returns:
    Tuple[float, float, float]: The precision, recall, and F1 score for ROUGE-N.
    """
    # Define the ROUGE scorer with the specified n-gram length
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    
    # Calculate the ROUGE-N score
    scores = scorer.score(actual_text, hallucinated_text)
    
    # Extract precision, recall, and F1 score
    precision = scores[f'rouge{n}'].precision
    recall = scores[f'rouge{n}'].recall
    f1_score = scores[f'rouge{n}'].fmeasure
    
    return precision, recall, f1_score

vocab = {}
right_answers = []
hallucinated_answers = []
knowledge_grams = []
question_grams = []
right_answer_grams = []
right_answer_grams2 = []
hallucinated_answer_grams = []
hallucinated_answer_grams2 = []
dataset = "truthfulqa" # truthfulqa halueval
metric_setting = "correct-incorrect" # correct-incorrect correct-correct incorrect-incorrect
svd_setting = "row" # row column
ngram = 2
vocab['<PAD>'] = 0

file_path = 'data/TruthfulQA.csv'
data = pd.read_csv(file_path)

if dataset == "halueval":
    for i, entry in enumerate(data):
        right_answers.append(entry.get("right_answer"))
        hallucinated_answers.append(entry.get("hallucinated_answer"))   
        knowledge_grams.append(generate_word_ngrams(entry.get("knowledge"), ngram))
        question_grams.append(generate_word_ngrams(entry.get("question"), ngram))
        right_answer_grams.append(generate_word_ngrams(entry.get("right_answer"), ngram))
        hallucinated_answer_grams.append(generate_word_ngrams(entry.get("hallucinated_answer"), ngram))

if dataset == "truthfulqa":
    # Process the dataset
    for index, row in data.iterrows():
        # Split the correct and incorrect answers into lists
        correct_answers_list = [answer.strip() for answer in row['Correct Answers'].split(';')]
        incorrect_answers_list = [answer.strip() for answer in row['Incorrect Answers'].split(';')]
        
        # Generate n-grams for the question
        question_ngram = generate_word_ngrams(row['Question'], ngram)
        
        if metric_setting == "correct-incorrect":
            # Pair each correct answer with each incorrect answer
            for correct_answer in correct_answers_list[:1]:
                for incorrect_answer in incorrect_answers_list[:1]:
                    right_answers.append(correct_answer)
                    hallucinated_answers.append(incorrect_answer)
                    question_grams.append(question_ngram)
                    right_answer_grams.append(generate_word_ngrams(correct_answer, ngram))
                    hallucinated_answer_grams.append(generate_word_ngrams(incorrect_answer, ngram))
        elif metric_setting == "correct-correct":
            # Pair each correct answer with another correct answer
            for i in range(len(correct_answers_list)):
                for j in range(i + 1, len(correct_answers_list)):
                    right_answers.append(correct_answers_list[i])
                    hallucinated_answers.append(correct_answers_list[j])
                    question_grams.append(question_ngram)
                    right_answer_grams.append(generate_word_ngrams(correct_answers_list[i], ngram))
                    hallucinated_answer_grams.append(generate_word_ngrams(correct_answers_list[j], ngram))
        elif metric_setting == "incorrect-incorrect":
            # Pair each incorrect answer with another incorrect answer
            for i in range(len(incorrect_answers_list)):
                for j in range(i + 1, len(incorrect_answers_list)):
                    right_answers.append(incorrect_answers_list[i])
                    hallucinated_answers.append(incorrect_answers_list[j])
                    question_grams.append(question_ngram)
                    right_answer_grams.append(generate_word_ngrams(incorrect_answers_list[i], ngram))
                    hallucinated_answer_grams.append(generate_word_ngrams(incorrect_answers_list[j], ngram))

cosine_similarities = []
procrustes_distances = []
cka_scores = []
rouge_scores = []

# New variable to store modified data
modified_data = []
for i in range(len(question_grams)):
    # Pad grams to the same size before SVD
    if metric_setting == "correct-incorrect":
        output = pad_grams([hallucinated_answer_grams[i], right_answer_grams[i]], ngram)
        hallucinated_answer_grams[i], right_answer_grams[i] = output[0], output[1]
    elif metric_setting == "correct-correct":
        output = pad_grams([hallucinated_answer_grams[i], right_answer_grams[i]], ngram)
        hallucinated_answer_grams[i], right_answer_grams[i] = output[0], output[1]

    # Perform SVD
    U, S, V = torch.svd(torch.tensor(right_answer_grams[i], dtype=torch.float))
    if svd_setting == "row":
        Xo = U
    elif svd_setting == "column":
        Xo = torch.mm(U, torch.diag(S))

    U, S, V = torch.svd(torch.tensor(hallucinated_answer_grams[i], dtype=torch.float))
    if svd_setting == "row":
        Xg = U
    elif svd_setting == "column":
        Xg = torch.mm(U, torch.diag(S))

    if metric_setting == "incorrect-incorrect":


    if metric_setting == "correct-correct":


    # print(Xo.shape, Xg.shape)
    # exit()

    # # Pad to the same size after SVD
    # Xo, Xg = pad_to_same_size(Xo, Xg)

    # Compute the cosine similarity between the two matrices
    cosine_similarity = torch.nn.functional.cosine_similarity(Xo, Xg, dim=1)
    cosine_similarities.append(torch.mean(cosine_similarity).item())

    # Compute the Procrustes analysis
    disparity = compute_procrustes_distances(Xo, Xg)
    procrustes_distances.append(disparity)

    # Compute the CKA similarity
    cka_score = compute_cka(Xo, Xg)
    cka_scores.append(cka_score)

    # Calculate the Rouge-N score between the right and hallucinated answers
    rouge_scores.append(calculate_rouge_n(right_answers[i], hallucinated_answers[i], ngram))

    # Store modified data in the new variable
    modified_entry = {
        'cosine': torch.mean(cosine_similarity).item(),
        'procrustes': disparity,
        'cka': cka_score,
        'right_grams': len(right_answer_grams[i]),
        'hallucinated_grams': len(hallucinated_answer_grams[i]),
        'question_grams': len(question_grams[i]),
        'rouge': rouge_scores[i]
    }
    modified_data.append(modified_entry)

print("-"*50)
print("Average Cosine Similarity:", np.mean(cosine_similarities))
print("Min Cosine Similarity:", np.min(cosine_similarities))
print("Max Cosine Similarity:", np.max(cosine_similarities))
print("Standard Deviation of Cosine Similarity:", np.std(cosine_similarities))
print("-"*50)
print("Average Procrustes Distance:", np.mean(procrustes_distances))
print("Min Procrustes Distance:", np.min(procrustes_distances))
print("Max Procrustes Distance:", np.max(procrustes_distances))
print("Standard Deviation of Procrustes Distance:", np.std(procrustes_distances))
print("-"*50)
print("Average CKA Score:", np.mean(cka_scores))
print("Min CKA Score:", np.min(cka_scores))
print("Max CKA Score:", np.max(cka_scores))
print("Standard Deviation of CKA Score:", np.std(cka_scores))
print("-"*50)
print(f"Average {ngram}-Rouge Score:", np.mean(rouge_scores))
print("Min Rouge Score:", np.min(rouge_scores))
print("Max Rouge Score:", np.max(rouge_scores))
print("Standard Deviation of Rouge Score:", np.std(rouge_scores))
print("-"*50)

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj

# Write the modified data to a new JSON file
with open('modified_data.json', 'w') as f:
    json.dump(modified_data, f, indent=4, default=convert_to_serializable)

