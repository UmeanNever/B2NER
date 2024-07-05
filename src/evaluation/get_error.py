import json
import csv

# Define the input and output file paths
input_file = 'output/MLV2-InternLM2-0s/20B+b2ner+bs128+23/eval-2/report/NER_zh/CLUENER.json'
output_file = 'output/evaluation/Errors_CLUENER.csv'

# Load the JSON data from the file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Open the output CSV file for writing
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Define the CSV writer
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['dataset', 'sentence', 'y_truth', 'y_predict'])

    # Iterate over the records in the JSON data
    for record in data['AuditAnyF1NotOne']['record']:
        # Extract the necessary information from each record
        dataset = record['json_data']['Dataset']
        sentence = record['json_data']['Instance']['sentence']
        y_truth = ';\n '.join(record['y_truth'])
        y_predict = ';\n '.join(record['y_pred'])

        # Write the extracted data to the CSV file
        writer.writerow([dataset, sentence, y_truth, y_predict])

print(f"Data has been successfully extracted and saved to {output_file}.")
