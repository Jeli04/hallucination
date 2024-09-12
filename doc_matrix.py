import torch 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import json 
import nltk
from nltk.util import ngrams
from scipy.spatial import procrustes
from tqdm import tqdm

'''
file_path = 'data/TruthfulQA.csv'
data = pd.read_csv(file_path)

# Ensure you have the required data files
nltk.download('punkt')

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
    return torch.tensor(input, dtype=torch.float).T # A tensor of shape (len(vocab), ngram)

def create_vocab(data, keys=['right_answer']):
    vocab = {}

    if "Correct Answers" in keys or "Incorrect Answers" in keys:
        for i, entry in data.iterrows():
            for key in keys:
                answer_list = entry[key].split(';')
                for answer in answer_list:
                    words = nltk.word_tokenize(answer)  # only gets the first
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


        
def compute_svd(input_tensor, k=2, svd_setting="row"):
    # Move input tensor to GPU if it's not already
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')
    
    # Perform SVD
    U, S, V = torch.linalg.svd(input_tensor, full_matrices=True)
    
    # Reduce dimensionality based on k
    U = U[:, :k]
    S = S[:k]
    V = V[:, :k]
    
    # Compute output based on svd_setting
    if svd_setting == "row":
        output = torch.mm(U, torch.diag(S))
    elif svd_setting == "col":
        output = V

    # Ensure the output is moved back to CPU for further processing if needed
    return output.to('cpu') if not input_tensor.is_cuda else output


right_vocab = create_vocab(data, keys=['Correct Answers'])
halu_vocab = create_vocab(data, keys=['Incorrect Answers'])

# Union the vocabularies
vocab = {**right_vocab, **halu_vocab}

real_answers = []
halucinated_answers = []
real_svd_input = []
halucinated_svd_input = []
ngram = 2
k = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ngram hash map creation
for index, row in data.iterrows():
    # for answer in row['Correct Answers'].split(';')[0]:
    #     real_answers.append(answer.strip())
    #     real_svd_input.append(prepare_svd_input(generate_word_ngrams(real_answers[-1], ngram), vocab, ngram).to(device))

    # for answer in row['Incorrect Answers'].split(';')[0]:
    #     halucinated_answers.append(answer.strip())
    #     halucinated_svd_input.append(prepare_svd_input(generate_word_ngrams(halucinated_answers[-1], ngram), vocab, ngram).to(device))
    answer = row['Correct Answers'].split(';')[0]
    real_answers.append(answer.strip())
    real_svd_input.append(prepare_svd_input(generate_word_ngrams(real_answers[-1], ngram), vocab, ngram).to(device))

    answer = row['Incorrect Answers'].split(';')[0]
    halucinated_answers.append(answer.strip())
    halucinated_svd_input.append(prepare_svd_input(generate_word_ngrams(halucinated_answers[-1], ngram), vocab, ngram).to(device))

# SVD computation
Xos = []
Xgs = []    
for i in tqdm(range(len(real_svd_input)), desc="Outer Loop Progress"):
    Xos.append(compute_svd(real_svd_input[i], k, "row"))
    Xgs.append(compute_svd(halucinated_svd_input[i], k, "row"))

Xos = torch.stack(Xos)  
Xgs = torch.stack(Xgs)  
Xs = torch.cat((Xos, Xgs))

# Create a matrix to hold the results
matrix = np.zeros((len(Xs), len(Xs)))

# Fill the upper triangular part of the matrix with the computed metrics
for i in tqdm(range(len(Xs)), desc="Outer Loop Progress"):
    for j in tqdm(range(i, len(Xs)), desc="Inner Loop Progress", leave=False):
        matrix[i, j] = compute_procrustes_distances(Xs[i], Xs[j])
        matrix[j, i] = matrix[i, j]  # Mirror the value to the lower triangular part

# Convert the torch matrix to numpy for plotting
# matrix_np = matrix.numpy()

# save the matrix to a file
np.save('procrustes_distances.npy', matrix)

print(matrix.shape)
'''
matrix = np.load('procrustes_distances.npy')
print("loaded")

# Plot the matrix as a heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, cmap='coolwarm')
plt.title("Metric Matrix for Concatenated x and y (PyTorch)")
plt.xlabel("Concatenated x and y")
plt.ylabel("Concatenated x and y")
# plt.show()

# Save the plot as an image
plt.savefig('heatmap.png', dpi=300)  # Higher DPI for better quality
plt.close()  # Close the plot to free up memory