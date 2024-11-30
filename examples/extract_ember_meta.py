import json

# File paths
# add a base path if needed
files = ["train_features_0.jsonl",
          "train_features_1.jsonl",
          "train_features_2.jsonl",
          "train_features_3.jsonl",
          "train_features_4.jsonl",
           "train_features_5.jsonl",
          "test_features.jsonl"
          ]
output_file = "extracted_data.jsonl"

# Initialize an empty list to store extracted records
extracted_data = []

# Read and process the JSONL file
for input_file in files:
    print(input_file
          )
    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                # Parse each line as a JSON object
                record = json.loads(line.strip())
                # Extract the required fields
                extracted_record = {
                    "appeared": record.get("appeared"),
                    "timestamp": record.get('header').get('coff').get("timestamp"),
                    "label": record.get('label')
                }
                extracted_data.append(extracted_record)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    # Write the extracted data to a new JSONL file
    with open(output_file, 'w') as outfile:
        for record in extracted_data:
            outfile.write(json.dumps(record) + "\n")

print(f"Extracted data saved to {output_file}")
