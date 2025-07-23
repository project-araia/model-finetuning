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

def is_variable(placeholder):
    """Returns True if the placeholder is a simple variable name."""
    return re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", placeholder) is not None

def separate_vars_and_exprs(text):
    """
    Splits all placeholders into variables and expressions.
    Returns a tuple: (variables, expressions)
    """
    variables = []
    expressions = []

    for placeholder in extract_placeholders(text):
        if is_variable(placeholder):
            variables.append(placeholder)
        else:
            expressions.append(placeholder)

    return variables, expressions

# --- Save generated data ---
def save_template(filename, mode, data):
    with open(filename, mode) as out_file:
        json.dump(data, out_file, indent=4)
    print(f"Data saved to {filename}")
