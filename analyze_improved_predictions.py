import json
import re

# Function equivalent to our improved extract_answer in custom_gsm8k_eval.py
def extract_answer(text):
    """Extract the answer from the text."""
    # First check for the standard GSM8K format with ####
    if '####' in text:
        match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
    
    # Look for "Therefore" or "Thus" statements with numbers at the end
    therefore_pattern = re.search(r'(?:Therefore|Thus|In conclusion|The answer is)[^.]*?(\d+(?:,\d+)*(?:\.\d+)?)[^.]*\.', text)
    if therefore_pattern:
        return therefore_pattern.group(1).replace(',', '')
    
    # Look for the last equation with an equal sign
    equations = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*=\s*(\$?\s*\d+(?:,\d+)*(?:\.\d+)?)', text)
    if equations:
        # Get the last equation result, remove $ if present
        last_result = equations[-1][1].replace('$', '').replace(',', '').strip()
        return last_result
    
    # Look for dollar amounts
    money_pattern = re.search(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if money_pattern:
        return money_pattern.group(1).replace(',', '')
    
    # Split by sentences and find the last sentence with a number
    sentences = re.split(r'[.!?]+', text)
    for sentence in reversed(sentences):
        number_pattern = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', sentence)
        if number_pattern:
            return number_pattern.group(1).replace(',', '')
    
    return None

# Path to predictions file
predictions_path = '/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k/gsm8k_predictions.jsonl'

# Load and analyze predictions
all_examples = []

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
        
        # Extract prediction answer with our improved function
        pred_answer = extract_answer(prediction)
        
        # Compare and categorize
        is_correct = pred_answer == ref_answer
        
        all_examples.append({
            'id': example_id,
            'question': question,
            'prediction': prediction,
            'reference': reference,
            'extracted_answer': pred_answer,
            'ref_answer': ref_answer,
            'is_correct': is_correct
        })

# Print summary
correct_examples = [ex for ex in all_examples if ex['is_correct']]
print(f"Total examples: {len(all_examples)}")
print(f"Correct examples: {len(correct_examples)}")
print(f"Accuracy: {len(correct_examples)/len(all_examples):.4f}")

# Print first 5 correct examples with their questions and answers
print("\n=== SAMPLE CORRECT EXAMPLES ===")
for i, ex in enumerate(correct_examples[:5]):
    print(f"\nExample {ex['id']}")
    print(f"Question: {ex['question'][:100]}...")
    print(f"Extracted answer: {ex['extracted_answer']}")
    print(f"Reference answer: {ex['ref_answer']}")

# Print a few incorrect examples
incorrect_examples = [ex for ex in all_examples if not ex['is_correct']]
print("\n=== SAMPLE INCORRECT EXAMPLES ===")
for i, ex in enumerate(incorrect_examples[:5]):
    print(f"\nExample {ex['id']}")
    print(f"Question: {ex['question'][:100]}...")
    print(f"Extracted answer: {ex['extracted_answer'] or 'None'}")
    print(f"Reference answer: {ex['ref_answer']}") 