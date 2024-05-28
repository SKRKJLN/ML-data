import uuid
import re
import time
from datasets import load_dataset

def parse_questions_answers(entry):
    """Parse questions and answers from an entry."""
    qa_pairs = []
    # Assuming questions are followed by their answers in the format "Question? Yes/No"
    qa_pattern = re.compile(r"(?P<question>.*?\?)\s*(?P<answer>(Yes|No))", re.IGNORECASE)
    
    matches = qa_pattern.findall(entry)
    for match in matches:
        qa_pairs.append({
            "id": str(uuid.uuid4()),
            "Question": match[0].strip(),
            "Answer": match[1].strip().capitalize()
        })
    return qa_pairs

def transform_dataset(data):
    """Transform the entire dataset."""
    all_qa_pairs = []
    for entry in data['input']:
        qa_pairs = parse_questions_answers(entry)
        all_qa_pairs.extend(qa_pairs)
    return all_qa_pairs

def save_to_json(data, output_file):
    """Save the transformed data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    output_file = 'reformatted_dataset.json'
    
    # Start the timer
    start_time = time.time()

    # Load the dataset
    dataset = load_dataset('AdaptLLM/finance-tasks', 'headline')

    # Transform the dataset
    transformed_data = transform_dataset(dataset)
    
    # Save the reformatted dataset
    save_to_json(transformed_data, output_file)
    
    # Stop the timer
    end_time = time.time()
    
    # Report statistics
    total_pairs = len(transformed_data)
    time_taken = end_time - start_time
    
    print(f"Total number of question-answer pairs extracted: {total_pairs}")
    print(f"Time taken for transformation: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
