import json
import os
import pandas as pd
from datasets import load_dataset

def create_dataset():
    dataset = [
        {
            "instruction": "Provide predictions for rainfall in Chicago for August 2025.",
            "input": "<precipitation-data>",
            "context": "<context-for-preciptation-data>",
            "output": "Heavy rainfall expected due to snowfall"
        },       
    ]
    
    # Convert to DataFrame and save as JSONL
    df = pd.DataFrame(dataset)
    output_path = "fine_tuning_dataset.jsonl"
    df.to_json(output_path, orient="records", lines=True)
    
    print(f"Dataset saved as {output_path}")

def load_local_dataset():
    dataset_path = "fine_tuning_dataset.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found. Please run create_dataset() first.")
    return load_dataset("json", data_files=dataset_path)["train"]

if __name__ == "__main__":
    create_dataset()
    dataset = load_local_dataset()
    print(dataset)
