from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Sample data (you will replace this with the actual outputs)
reference_output = """The temperature at grid R160C290 is projected to increase by 9.69769001\u00b0F by the end of the century under RCP 8.5. This places it in the 76.5th percentile for warming within the state and 88.25th percentile nationally."""

base_output = """The average annual temperature increase in the area from 1995 to 2050 is 4.2 degrees Celsius. This is slightly higher than the average temperature increase in the area from 1995 to 2050. The average temperature increase in the area from 1995 to 2050 is 4.2 degrees Celsius. This is slightly higher than the average temperature increase in the area from 1995 to 2050. The average temperature increase in the area from 1995 to 2050 is 4.2 degrees Celsius. This is slightly higher than the average temperature increase in the area from 1995"""

fine_tuned_output = """The temperature at grid R160C290 is projected to increase by 9.69769001°F by the end of the century under RCP 8.5. This places it in the 76.5th percentile for warming within the state and 88.25th percentile nationally"""

# Get similarity scores
base_score, fine_tuned_score = get_similarity_score(reference_output, base_output, fine_tuned_output)

# Display results
print(f"Base Model Similarity Score: {base_score:.4f}")
print(f"Fine-tuned Model Similarity Score: {fine_tuned_score:.4f}")

