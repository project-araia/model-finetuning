import re
import json
import pandas as pd
import climparser
import templateparser

# --- Load climate dataset ---
climate_df = pd.read_csv("FullData.csv")

# --- Load templates with only question/answer ---
with open("Templates.json", "r") as f:
    templates = json.load(f)

selected_locations = ["Cook, IL", "Loudoun, VA"]
compared_locations = ["King, WA"]
generated_entries = []

for template in templates:
    question_template = template["question"]
    answer_template = template["answer"]

    # Dynamically extract metadata from question & answer
    template_vars = templateparser.extract_variables(question_template, answer_template)
    placeholders = templateparser.extract_placeholders(question_template, answer_template)
    key_pattern_str = templateparser.build_key_pattern(template_vars)
    key_pattern = re.compile(key_pattern_str)
    value_map = templateparser.build_extract_map(placeholders, template_vars)

    is_comparison = template_vars.get("is_comparative", False)

    if is_comparison:
        # Comparison case
        for loc1 in selected_locations:
            for loc2 in compared_locations:

                county1, state1 = map(str.strip, loc1.split(","))
                county2, state2 = map(str.strip, loc2.split(","))

                data1 = climparser.query_mean(climate_df, county1, state1)
                data2 = climparser.query_mean(climate_df, county2, state2)

                matched_keys1 = [k for k in data1.keys() if key_pattern.search(k)]
                matched_keys2 = [k for k in data2.keys() if key_pattern.search(k)]

                filled_vars = {
                    **template_vars,
                    "location": loc1,
                    "location2": loc2
                }

                # Extract values
                for varname, substring in value_map.items():
                    if varname.endswith("_loc2"):
                        for key in matched_keys2:
                            if substring in key:
                                filled_vars[varname] = data2[key]
                                break
                    else:
                        for key in matched_keys1:
                            if substring in key:
                                filled_vars[varname] = data1[key]
                                break

                # Compute comparison phrase if applicable
                if "seas_hist" in filled_vars and "seas_hist_loc2" in filled_vars:
                    v1 = filled_vars["seas_hist"]
                    v2 = filled_vars["seas_hist_loc2"]
                    if v1 > v2:
                        comparison = "warmer"
                    elif v1 < v2:
                        comparison = "cooler"
                    else:
                        comparison = "about the same"
                    filled_vars["comparison"] = comparison
                else:
                    continue

                required_vars = set(re.findall(r"{(.*?)}", question_template + answer_template))
                if not required_vars.issubset(filled_vars):
                    continue

                question = question_template.format(**filled_vars)
                answer = answer_template.format(**filled_vars)

                entry = {
                    "user": question,
                    "input": {
                        "location1_data": {k: data1[k] for k in matched_keys1},
                        "location2_data": {k: data2[k] for k in matched_keys2}
                    },
                    "assistant": answer
                }
                generated_entries.append(entry)

    else:
        # Single location case
        for loc in selected_locations:
            county, state = map(str.strip, loc.split(","))
            data = climparser.query_mean(climate_df, county, state)
            matched_keys = [k for k in data.keys() if key_pattern.search(k)]

            filled_vars = {
                **template_vars,
                "location": loc
            }

            for varname, substring in value_map.items():
                for key in matched_keys:
                    if substring in key:
                        filled_vars[varname] = data[key]
                        break

            required_vars = set(re.findall(r"{(.*?)}", question_template + answer_template))
            if not required_vars.issubset(filled_vars):
                continue

            question = question_template.format(**filled_vars)
            answer = answer_template.format(**filled_vars)

            entry = {
                "user": question,
                "input": {k: data[k] for k in matched_keys},
                "assistant": answer
            }
            generated_entries.append(entry)

# --- Save generated data ---
with open("Training.json", "w") as out_file:
    json.dump(generated_entries, out_file, indent=4)

print("Training data generated and saved as Training.json")
