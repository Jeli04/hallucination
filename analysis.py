import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

file_path = "modified_data.json"
with open(file_path, 'r') as f:
    data = json.load(f)

def plot_data(x, y, x_title, y_title, title):
    # Plot the number of tokens vs. cosine similarity
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.show()

def plot_multi_data(x, y, x_labels, x_title, y_title, title):
    plt.figure(figsize=(10, 6))

    for i in range(len(x)):
        plt.scatter(x[i], y, label=x_labels[i], alpha=0.7)

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=12, title='Legend')
    plt.show()


def plot_heat_map(x, y, x_title, y_title, title, num_ticks=4):

    # Prepare DataFrame
    data = {
        y_title: y,
        x_title: x
    }

    df = pd.DataFrame(data)

    # Create heatmap
    plt.figure(figsize=(14, 10))
    heatmap_data = pd.pivot_table(df, index=y_title, values=x_title, aggfunc=np.mean)
    heatmap = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', cbar=True)

    # Set the x-axis to display only a few ticks
    x_ticks = np.linspace(0, len(heatmap_data.columns) - 1, num=4, dtype=int)
    x_ticklabels = [heatmap_data.columns[i] for i in x_ticks]
    heatmap.set_xticks(x_ticks)
    heatmap.set_xticklabels(x_ticklabels, rotation=45, ha='right', fontsize=10)

    # Set the y-axis to display only a few ticks
    y_ticks = np.linspace(0, len(heatmap_data.index) - 1, num=4, dtype=int)
    y_ticklabels = [heatmap_data.index[i] for i in y_ticks]
    heatmap.set_yticks(y_ticks)
    heatmap.set_yticklabels(y_ticklabels, rotation=0, fontsize=10)

    # Invert y-axis to correct the orientation
    plt.gca().invert_yaxis()

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

cosine = []
procrustes = []
right_ngrams = []
hallucinated_ngrams = []
knowledge_ngrams = []
rouge_scores = []
cka = []
for entry in data:
    right_ngrams.append(entry['right_grams'])
    hallucinated_ngrams.append(entry['hallucinated_grams'])
    # knowledge_ngrams.append(entry['knowledge_grams'])
    cosine.append((entry['cosine'] + 1) / 2)
    procrustes.append(entry['procrustes'])   
    rouge_scores.append(entry['rouge'][-1])
    cka.append(entry['cka'])


# No ROUGE Scores
# plot_data(right_ngrams, cosine, "Number of Grams", "Cosine Similarity", "Number of Right Answer Grams vs. Cosine Similarity")
# plot_data(right_ngrams, procrustes, "Number of Grams", "Procrustes Distance", "Number of Right Answer Grams vs. Procrustes Distance")

# plot_data(hallucinated_ngrams, cosine, "Number of Grams", "Cosine Similarity", "Number of Hallucinated Answer Grams vs. Cosine Similarity")
# plot_data(hallucinated_ngrams, procrustes, "Number of Grams", "Procrustes Distance", "Number of Hallucinated Answer Grams vs. Procrustes Distance")

# plot_data(knowledge_ngrams, cosine, "Number of Grams", "Cosine Similarity", "Number of Knowledge Answer Grams vs. Cosine Similarity")
# plot_data(knowledge_ngrams, procrustes, "Number of Grams", "Procrustes Distance", "Number of Knowledge Answer Grams vs. Procrustes Distance")

# ROUGE scores
# plot_data(rouge_scores, cosine, "Rouge F1", "Cosine Similarity", "Right Answers vs.Cosine Similarity of Knowledge Grams")
# plot_data(rouge_scores, procrustes, "Rouge F1", "Procrustes", "Right Answers vs.Procrustes Distance of Knowledge Grams")
# plot_heat_map(rouge_scores, procrustes, "Rouge F1", "Procrustes", "Right Answers vs.Procustes Distance of Knowledge Grams")

# Cosine Similarity vs. Procrustes Distance
plot_data(procrustes, cosine, "Procrustes Distance", "Cosine Similarity", "Procrustes Distance vs. Cosine Similarity Between Hallucinations and Non-Hallucinations (2 Grams)")
# plot_data(procrustes, cka, "Procrustes Distance", "CKA", "Procrustes Distance vs. CKA Between Hallucinations and Non-Hallucinations (2 Grams)")
# plot_data(cka, cosine, "CKA", "Cosine Similarity", "CKA vs. Cosine Similarity Between Hallucinations and Non-Hallucinations (2 Grams)")

plot_multi_data([cosine, procrustes], rouge_scores, ["cosine", "procurstes"], "Metric", "Rouge F1", "Cosine Similarity & Procrustes vs. ROUGE-2 Score")
# plot_multi_data([cosine, procrustes, cka], rouge_scores, ["cosine", "procrustes", "cka"], "Metric", "Rouge F1", "Cosine Similarity & Procrustes & CKA vs. ROUGE-2 Score")