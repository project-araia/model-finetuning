from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import json

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate cosine similarity score
def get_similarity_score(reference, base_output, fine_tuned_output):
    # Generate embeddings for each response
    reference_embedding = model.encode([reference])
    base_embedding = model.encode([base_output])
    fine_tuned_embedding = model.encode([fine_tuned_output])

    # Compute cosine similarity scores
    base_similarity = cosine_similarity(reference_embedding, base_embedding)[0][0]
    fine_tuned_similarity = cosine_similarity(reference_embedding, fine_tuned_embedding)[0][0]

    return base_similarity, fine_tuned_similarity

def extract_assistant_responses(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regular expression to extract Assistant responses
    pattern = re.compile(r"### Assistant:\s*(.*?)(?=\n###|<\|end_of_text\|>|$)", re.S)
    responses = pattern.findall(text)

    pattern_str = "---------------------------------------------------------------------------------------------"

    # Clean up unwanted headers and separators
    cleaned_responses = []
    for response in responses:
        split_response = response.split(pattern_str)
        cleaned_responses.append(split_response[0])

    return cleaned_responses

def extract_assistant_responses_json(file_path):
    # Load the JSON data from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract assistant responses
    responses = [entry['assistant'] for entry in data if 'assistant' in entry]

    return responses

# Path to your text file
base_responses = extract_assistant_responses('../runs/training/outputs-march-2025/output_base_model_llama3.1_8b.txt')
finetuned_responses = extract_assistant_responses('../runs/training/outputs-march-2025/output_finetuned_model.txt') 
ref_responses = extract_assistant_responses_json('../datasets/Testing/AnnualTemperatureMaximum/WithoutInputContext.json')

score_count = min(len(base_responses),len(finetuned_responses))

base_score = 0.
fine_tuned_score = 0.

for i in range(score_count):
    base_output = base_responses[i].strip("\n")
    fine_tuned_output = finetuned_responses[i].strip("\n")
    reference_output = ref_responses[i]

    # Get similarity scores
    scores = get_similarity_score(reference_output, base_output, fine_tuned_output)

    base_score = base_score + scores[0]
    fine_tuned_score = fine_tuned_score + scores[1]


# Display results
print(f"Base Model Similarity Score: {base_score/score_count:.4f}")
print(f"Fine-tuned Model Similarity Score: {fine_tuned_score/score_count:.4f}")
