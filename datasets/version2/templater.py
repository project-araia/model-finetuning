import re
import json

# --- Load filename in json format
def load_template(filename):
    with open(filename, "r") as f:
        templates = json.load(f)
    return templates

# --- Extract all {variable} placeholders from a string ---
def extract_placeholders(text):
    return re.findall(r"{(.*?)}", text)

# --- Save generated data ---
def save_template(filename, mode, data):
    with open(filename, mode) as out_file:
        json.dump(data, out_file, indent=4)
    print(f"Data saved to {filename}")
