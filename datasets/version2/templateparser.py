import json
import re

def extract_variables(question, answer):
    question_lower = question.lower()

    # Detect temperature type
    if "maximum" in question_lower:
        temperature_type = "maximum"
    elif "minimum" in question_lower:
        temperature_type = "minimum"
    else:
        temperature_type = None

    # Detect season
    seasons = ["spring", "summer", "autumn", "fall", "winter"]
    season = None
    for s in seasons:
        if s in question_lower:
            season = "autumn" if s == "fall" else s
            break

    # Detect scenario
    scenario = None
    m = re.search(r"rcp\s?(\d+)", question_lower)
    if m:
        scenario = m.group(1)

    # Detect time period
    time_period = None
    if "by mid" in question_lower or "mid-century" in question_lower:
        time_period = "mid"
    elif "by end" in question_lower or "end-century" in question_lower:
        time_period = "end"

    # Detect comparison
    is_comparative = any(word in question_lower for word in ["compare", "vs", "than"])

    # Detect variable type
    if "cooling degree" in question_lower:
        variable_type = "cooling_degree_days"
    elif "heating degree" in question_lower:
        variable_type = "heating_degree_days"
    elif "precipitation" in question_lower or "rainfall" in question_lower:
        variable_type = "precipitation"
    elif "fire weather" in question_lower or "fire risk" in question_lower:
        variable_type = "fire_weather_index"
    elif "temperature" in question_lower:
        variable_type = "temperature"
    else:
        variable_type = None

    return {
        "temperature_type": temperature_type,
        "season": season,
        "scenario": scenario,
        "time_period": time_period,
        "is_comparative": is_comparative,
        "variable_type": variable_type
    }

def extract_placeholders(question, answer):
    placeholders_q = re.findall(r"\{([^}]+)\}", question)
    placeholders_a = re.findall(r"\{([^}]+)\}", answer)
    placeholders = sorted(set(placeholders_q + placeholders_a))
    return placeholders

def build_key_pattern(variables):
    var_type = (variables.get("variable_type") or "").lower()
    season = variables.get("season", "").lower() if variables.get("season") else None
    temp_map = {
        "maximum": "max",
        "minimum": "min",
        "average": "avg"
    }
    temperature_type = variables.get("temperature_type")
    temp_short = temp_map.get(temperature_type.lower()) if temperature_type else None

    if var_type == "temperature" and temp_short:
        # Matches annual and seasonal temperature keys
        # Examples: tempmaxann_hist, tempmax_seas_hist_summer
        if season:
            # season can be spring, summer, autumn, winter
            return fr"temp{temp_short}(_seas_.*{season})"
        else:
            return r"temp(max|min|avg).*"
    elif var_type == "precipitation":
        # Precipitation keys have "precipann_" or "precipdaily_" prefixes with optional seasons
        if season:
            # Match daily or annual precip with season, e.g., precipdaily_histmean_winter, precipann_mid85_hist
            return fr"(precipann_.*{season}|precipdaily_.*{season})"
        else:
            return r"(precipann_.*|precipdaily_.*)"
    elif var_type == "cooling_degree_days":
        return r"cdd.*"
    elif var_type == "heating_degree_days":
        return r"hdd.*"
    elif var_type == "fire_weather_index":
        # Keys can be wildfire_<season>_<scenario> or FWI_Bins_<season>_<metric>
        # season: autumn, spring, summer, winter (if provided)
        # Also handle variants like FWIBins_HistSum_95 (history summer 95th percentile)
        if season:
            return fr"(wildfire_.*{season}|FWI_Bins_.*{season}|FWIBins_.*{season})"
        else:
            return r"(wildfire_.*|FWI_Bins_.*|FWIBins_.*)"
    else:
        return None

def build_extract_map(placeholders, variables):
    extract_map = {}
    for ph in placeholders:
        if ph == "seas_hist_loc2" and variables.get("is_comparative"):
            extract_map[ph] = "seas_hist"
        else:
            extract_map[ph] = ph
    return extract_map
