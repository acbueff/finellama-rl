#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare GSM8K data by creating a validation set from the training data.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GSM8K data with a validation split")
    
    parser.add_argument("--train_path", type=str, 
                        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json",
                        help="Path to the GSM8K training data")
    
    parser.add_argument("--test_path", type=str, 
                        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json",
                        help="Path to the GSM8K test data")
    
    parser.add_argument("--output_dir", type=str, 
                        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k",
                        help="Output directory for processed data")
    
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Size of validation set as a fraction of training data")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} examples to {file_path}")

def split_train_val(data: List[Dict[str, Any]], val_size: float, seed: int) -> tuple:
    """Split data into training and validation sets."""
    random.seed(seed)
    random.shuffle(data)
    
    val_count = int(len(data) * val_size)
    
    val_data = data[:val_count]
    train_data = data[val_count:]
    
    return train_data, val_data

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original training data
    print(f"Loading training data from {args.train_path}")
    train_data = load_jsonl(args.train_path)
    print(f"Loaded {len(train_data)} training examples")
    
    # Split into new train and validation sets
    new_train_data, val_data = split_train_val(train_data, args.val_size, args.seed)
    print(f"Split data into {len(new_train_data)} training and {len(val_data)} validation examples")
    
    # Save new splits
    new_train_path = os.path.join(args.output_dir, "gsm8k_train_split.json")
    val_path = os.path.join(args.output_dir, "gsm8k_val.json")
    
    save_jsonl(new_train_data, new_train_path)
    save_jsonl(val_data, val_path)
    
    # Check if test data exists and report
    if os.path.exists(args.test_path):
        test_data = load_jsonl(args.test_path)
        print(f"Test set contains {len(test_data)} examples")
    
    print("\nData preparation complete!")
    print(f"Training set: {new_train_path} ({len(new_train_data)} examples)")
    print(f"Validation set: {val_path} ({len(val_data)} examples)")
    if os.path.exists(args.test_path):
        print(f"Test set: {args.test_path} ({len(test_data)} examples)")

if __name__ == "__main__":
    main() 