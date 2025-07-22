import json
import re

def extract_variables(question, answer):
    question_lower = question.lower()
    
    # temperature_type detection
    if "maximum" in question_lower:
        temperature_type = "maximum"
    elif "minimum" in question_lower:
        temperature_type = "minimum"
    else:
        temperature_type = None
    
    # season detection
    seasons = ["spring", "summer", "autumn", "fall", "winter"]
    season = None
    for s in seasons:
        if s in question_lower:
            season = "autumn" if s == "fall" else s
            break

    # scenario detection
    scenario = None
    m = re.search(r"rcp\s?(\d+)", question_lower)
    if m:
        scenario = m.group(1)
    
    # time_period detection
    time_period = None
    if "by mid" in question_lower:
        time_period = "mid"
    elif "by end" in question_lower:
        time_period = "end"

    vars_dict = {
        "temperature_type": temperature_type,
        "season": season
    }
    if scenario:
        vars_dict["scenario"] = scenario
    if time_period:
        vars_dict["time_period"] = time_period

    # Detect if question is comparative (mentions 'compare', 'vs', 'than')
    if any(word in question_lower for word in ["compare", "vs", "than"]):
        vars_dict["is_comparative"] = True
    else:
        vars_dict["is_comparative"] = False

    return vars_dict

def extract_placeholders(question, answer):
    placeholders_q = re.findall(r"\{([^}]+)\}", question)
    placeholders_a = re.findall(r"\{([^}]+)\}", answer)
    placeholders = sorted(set(placeholders_q + placeholders_a))
    return placeholders

def build_key_pattern(variables):
    temp_map = {
        "maximum": "max",
        "minimum": "min",
        "average": "avg"
    }
    temp_short = temp_map.get(variables.get("temperature_type"))
    season = variables.get("season")
    if temp_short and season:
        return f"temp{temp_short}.*{season}"
    else:
        return None

def build_extract_map(placeholders, variables):
    extract_map = {}
    for ph in placeholders:
        # Special handling for comparison placeholders
        if ph == "seas_hist_loc2" and variables.get("is_comparative"):
            # Map explicitly as in your example
            extract_map[ph] = "seas_hist"
        else:
            extract_map[ph] = ph
    return extract_map
