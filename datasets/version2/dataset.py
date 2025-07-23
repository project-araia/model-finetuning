import climparser
import templater

# --- Load climate dataset ---
climate_df = climparser.load_dataset("FullData.csv")

# --- Load templates with only question/answer ---
chat_templates = templater.load_template("Templates.json")

selected_locations = ["Cook, IL"]#, "Loudoun, VA"]
compared_locations = ["King, WA"]
generated_entries = []

for template in chat_templates:
    question_template = template["question"]
    answer_template = template["answer"]

    # Extract placeholders in the answer template
    template_vars = templater.extract_placeholders(question_template + answer_template)
    template_dict = {}
    input_dict = {}

    for loc in selected_locations:

        if "location" in template_vars:
            template_dict["location"] = loc

        if "compared_location" in template_vars:
            for loc2 in compared_locations:

                template_dict["compared_location"] = loc2

                county1, state1 = map(str.strip, loc.split(","))
                county2, state2 = map(str.strip, loc2.split(","))

                loc_data1 = climparser.query_mean(climate_df, county1, state1)
                loc_data2 = climparser.query_mean(climate_df, county2, state2)

                for key, value in loc_data1.items():
                    if key in template_vars:              
                        template_dict[key] = value
                        input_dict[key] = value

                for key, value in loc_data2.items():
                    if key+"_loc2" in template_vars:
                        template_dict[key+"_loc2"] = value
                        input_dict[key+"_loc2"] = value
                
                question = question_template.format(**template_dict)
                answer = answer_template.format(**template_dict)

                entry = {
                    "user": question,
                    "input": input_dict.copy(),
                    "assistant": answer
                }
                generated_entries.append(entry)

 
        else:
            county, state = map(str.strip, loc.split(","))
            loc_data = climparser.query_center(climate_df, county, state)

            for key, value in loc_data.items():
                if key in template_vars:
                    template_dict[key] = value
                    input_dict[key] = value

            question = question_template.format(**template_dict)
            answer = answer_template.format(**template_dict)

            entry = {
                "user": question,
                "input": input_dict.copy(),
                "assistant": answer
            }
            generated_entries.append(entry)

templater.save_template("Training.json", "w", generated_entries)
