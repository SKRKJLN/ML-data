import re
import json
import time
import uuid
from datasets import load_dataset

def parse_qa_pairs(entry):
    """
    Parses the input text to extract question-answer pairs.
    """
    # Regex pattern to match question-answer pairs
    qa_pattern = re.compile(r'(.*?)(Yes|No)\s')
    pairs = re.findall(qa_pattern, entry)
    return pairs

def save_to_json(pairs, filename='reformatted_dataset.json'):
    """
    Saves the extracted question-answer pairs to a JSON file with the specified format.
    """
    data = []
    for question, answer in pairs:
        data.append({
            "id": str(uuid.uuid4()),
            "Question": question.strip(),
            "Answer": answer.strip()
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Load the dataset
dataset = load_dataset('AdaptLLM/finance-tasks', 'Headline')

# Use the 'test' split
data_split = 'test'

start_time = time.time()

all_qa_pairs = []
for entry in dataset[data_split]:
    input_text = entry['input']
    pairs = parse_qa_pairs(input_text)
    all_qa_pairs.extend(pairs)

# Save the results to a JSON file
save_to_json(all_qa_pairs)

end_time = time.time()
time_taken = end_time - start_time

# Reporting statistics
total_pairs = len(all_qa_pairs)
print(f"Total question-answer pairs extracted: {total_pairs}")
print(f"Time taken for the process: {time_taken:.2f} seconds")

# Schema Documentation (Printed for clarity)
schema = {
    "id": "<unique_identifier>",
    "Question": "<question_text>",
    "Answer": "<answer_text>"
}
print("Schema of the reformatted dataset:", json.dumps(schema, indent=4))