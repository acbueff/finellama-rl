import json
import random
import re

# Function to extract numbers from prediction
def extract_answer(text):
    # Check for #### format first
    if '####' in text:
        match = re.search(r'####\s*(\d+)', text)
        return match.group(1) if match else None
    
    # Try to find equations (number = number)
    equations = re.findall(r'(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', text)
    if equations:
        return equations[-1][1]
    
    # Try to find money amounts ($number)
    money_pattern = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if money_pattern:
        return money_pattern.group(1).replace(',', '')
    
    # Try to find numbers in the last sentence
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        last_sentence = sentences[-1].strip()
        number_pattern = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', last_sentence)
        if number_pattern:
            return number_pattern.group(1).replace(',', '')
    
    return None

# Path to predictions file
predictions_path = '/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k/gsm8k_predictions.jsonl'

# Load and analyze predictions
correct_examples = []
incorrect_examples = []

with open(predictions_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        question = data['question']
        prediction = data['prediction']
        reference = data['reference']
        example_id = data['id']
        
        # Extract reference answer
        if '#### ' in reference:
            ref_answer = reference.split('#### ')[1]
        else:
            # Fallback if reference doesn't have #### format
            ref_answer = extract_answer(reference)
        
        # Extract prediction answer
        pred_answer = extract_answer(prediction)
        
        # Compare
        if pred_answer == ref_answer:
            correct_examples.append((example_id, question, prediction, reference, pred_answer, ref_answer))
        else:
            incorrect_examples.append((example_id, question, prediction, reference, pred_answer, ref_answer))

# Print summary
print(f"Total examples: {len(correct_examples) + len(incorrect_examples)}")
print(f"Correct examples: {len(correct_examples)}")
print(f"Incorrect examples: {len(incorrect_examples)}")
print(f"Accuracy: {len(correct_examples)/(len(correct_examples) + len(incorrect_examples)):.4f}")

# Show some examples
print("\n=== CORRECT EXAMPLES ===")
for i, (id, q, p, r, pa, ra) in enumerate(random.sample(correct_examples, min(2, len(correct_examples)))):
    print(f"\nExample {id}")
    print(f"Question: {q[:100]}...")
    print(f"Extracted answer: {pa}")
    print(f"Reference answer: {ra}")

print("\n=== INCORRECT EXAMPLES ===")
for i, (id, q, p, r, pa, ra) in enumerate(random.sample(incorrect_examples, min(2, len(incorrect_examples)))):
    print(f"\nExample {id}")
    print(f"Question: {q[:100]}...")
    print(f"Extracted answer: {pa}")
    print(f"Reference answer: {ra}") 