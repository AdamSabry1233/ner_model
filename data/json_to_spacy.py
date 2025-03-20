import json
import os

# Define the file paths
input_file = "data/spacy_ner_expense.json"
output_file = "data/ner_dataset.json"

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' was not found.")
    exit()

# Load JSON dataset
with open(input_file, "r", encoding="utf-8") as file:
    try:
        json_data = json.load(file)  # Load JSON file
        
        # Extract "features" if it exists
        if isinstance(json_data, dict) and "features" in json_data:
            json_data = json_data["features"]  # Get list of expense records
        else:
            print("Error: JSON does not contain a valid 'features' list.")
            json_data = []

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        json_data = []

def prepare_ner_data(json_data):
    training_data = []

    for record in json_data:
        if not isinstance(record, dict) or "properties" not in record:
            print(f"Skipping invalid record: {record}")
            continue

        props = record["properties"]

        # Construct sentence for NER training
        sentence = (f"OBJECTID {props.get('OBJECTID', 'UNKNOWN')}. "
                    f"On {props.get('Date', 'UNKNOWN')}, {props.get('Agency', 'UNKNOWN')} conducted "
                    f"{props.get('Service', 'UNKNOWN')}. The expense of ${props.get('Amount', 0)} "
                    f"was categorized as {props.get('Spending_Description', 'UNKNOWN')} under "
                    f"{props.get('Fund', 'UNKNOWN')}. Vendor: {props.get('Vendor_Name', 'UNKNOWN')}. "
                    f"GlobalID: {props.get('GlobalID', 'UNKNOWN')}. Date Accuracy: {props.get('Date_Accuracy', 'UNKNOWN')}.")

        entities = []
        label_mapping = {
            "Date": "DATE",
            "Agency": "AGENCY",
            "Service": "SERVICE",
            "Spending_Description": "DESCRIPTION",
            "Fund": "FUND",
            "Amount": "AMOUNT",
            "Vendor_Name": "VENDOR",
            "name": "VENDOR",
            "OBJECTID": "OBJECTID",
            "GlobalID": "GLOBALID",
            "Date_Accuracy": "DATE_ACCURACY"
        }

        # Finding entity locations dynamically
        for key, label in label_mapping.items():
            value = str(props.get(key, "UNKNOWN"))
            if value in sentence:
                start = sentence.index(value)
                end = start + len(value)
                # Ensure OBJECTID is only added once
                if label == "OBJECTID" and any(ent[2] == "OBJECTID" for ent in entities):
                    continue
                if (start, end, label) not in entities:
                    entities.append((start, end, label))

        training_data.append((sentence, {"entities": entities}))

    return training_data

# Process the extracted data
if json_data:
    ner_data = prepare_ner_data(json_data)

    # Save the NER dataset
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(ner_data, outfile, indent=4)

    print(f"NER dataset prepared! Saved to {output_file}")
