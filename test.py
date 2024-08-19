import pandas as pd

# Read the CSV file
file_path = 'data/TruthfulQA.csv'
data = pd.read_csv(file_path)

# Print the first 5 rows of the DataFrame
print(data.head())

# Get the column names
column_names = data.columns.tolist()
print(column_names)

questions = data['Question']
right_answers = data['Correct Answers']
hallucination_answers = data['Incorrect Answers']

print(len(right_answers))
print(len(hallucination_answers))