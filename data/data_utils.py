"""
Data utilities for loading and processing the FineMath dataset.
"""

import os
import json
import logging
import random
from typing import Dict, List, Union, Tuple, Optional, Any

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class MathDatasetProcessor:
    """
    Class for loading and processing mathematical datasets for model training and evaluation.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: str = "Solve the following math problem step by step:\n{problem}\n\nSolution:",
        seed: int = 42
    ):
        """
        Initialize the dataset processor.
        
        Args:
            tokenizer: Tokenizer to use for encoding the data
            max_length: Maximum sequence length
            prompt_template: Template for formatting problems
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def load_dataset(
        self,
        dataset_name: str = None,
        dataset_path: str = None,
        split: str = None,
        subset: str = None,
        data_files: Union[str, List[str]] = None,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load a dataset from Hugging Face datasets or local files.
        
        Args:
            dataset_name: Name of the dataset in Hugging Face datasets
            dataset_path: Local path to dataset
            split: Dataset split (train, validation, test)
            subset: Dataset subset (if applicable)
            data_files: Files to load data from (if loading from local files)
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        if dataset_name:
            logger.info(f"Loading dataset {dataset_name} (split={split}, subset={subset})")
            dataset = load_dataset(dataset_name, subset, split=split)
        elif dataset_path:
            logger.info(f"Loading dataset from {dataset_path}")
            if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            else:
                dataset = load_dataset(dataset_path, data_files=data_files, split=split or 'train')
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")
        
        # Limit number of samples if requested
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    def prepare_dataset(
        self,
        dataset: Dataset,
        problem_key: str = "question",
        solution_key: str = "answer",
        apply_template: bool = True,
        add_special_tokens: bool = True,
        return_tensors: bool = True
    ) -> Dataset:
        """
        Prepare a dataset for training or evaluation.
        
        Args:
            dataset: Input dataset
            problem_key: Key for the problem text in the dataset
            solution_key: Key for the solution text in the dataset
            apply_template: Whether to apply the prompt template
            add_special_tokens: Whether to add special tokens in tokenization
            return_tensors: Whether to return PyTorch tensors
            
        Returns:
            Processed dataset
        """
        def format_example(example):
            problem = example[problem_key]
            solution = example[solution_key] if solution_key in example else ""
            
            if apply_template:
                prompt = self.prompt_template.format(problem=problem)
            else:
                prompt = problem
                
            # Create full text (prompt + solution)
            full_text = prompt + " " + solution if solution else prompt
            
            # Tokenize
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length" if return_tensors else False,
                return_tensors="pt" if return_tensors else None,
                add_special_tokens=add_special_tokens
            )
            
            # Tokenize prompt separately to identify which tokens are prompt vs. solution
            prompt_tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=add_special_tokens
            )
            prompt_length = len(prompt_tokenized["input_ids"])
            
            # Create labels (set to -100 for prompt tokens to ignore in loss)
            if return_tensors:
                labels = tokenized["input_ids"].clone()
                labels[:, :prompt_length] = -100
            else:
                labels = tokenized["input_ids"].copy()
                labels[:prompt_length] = [-100] * prompt_length
                
            example.update({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
                "prompt": prompt,
                "solution": solution,
                "prompt_length": prompt_length
            })
            return example
            
        logger.info(f"Preparing dataset with {len(dataset)} examples")
        processed_dataset = dataset.map(
            format_example,
            desc="Formatting dataset",
            load_from_cache_file=False
        )
        
        return processed_dataset
    
    def prepare_for_ppo(
        self,
        dataset: Dataset,
        problem_key: str = "question",
        reference_solution_key: str = "answer",
        apply_template: bool = True
    ) -> Dataset:
        """
        Prepare a dataset specifically for PPO training.
        
        Args:
            dataset: Input dataset
            problem_key: Key for the problem text in the dataset
            reference_solution_key: Key for the reference solution
            apply_template: Whether to apply the prompt template
            
        Returns:
            Dataset prepared for PPO
        """
        def format_for_ppo(example):
            problem = example[problem_key]
            reference = example[reference_solution_key] if reference_solution_key in example else ""
            
            if apply_template:
                prompt = self.prompt_template.format(problem=problem)
            else:
                prompt = problem
                
            example.update({
                "prompt": prompt,
                "reference": reference
            })
            return example
            
        logger.info(f"Preparing dataset for PPO with {len(dataset)} examples")
        return dataset.map(
            format_for_ppo,
            desc="Formatting for PPO",
            load_from_cache_file=False
        )
    
    def prepare_for_grpo(
        self,
        dataset: Dataset,
        problem_key: str = "question",
        reference_solution_key: str = "answer",
        apply_template: bool = True
    ) -> Dataset:
        """
        Prepare a dataset specifically for GRPO training.
        Almost identical to PPO preparation, but we keep this separate in case
        we need to add GRPO-specific preprocessing later.
        
        Args:
            dataset: Input dataset
            problem_key: Key for the problem text in the dataset
            reference_solution_key: Key for the reference solution
            apply_template: Whether to apply the prompt template
            
        Returns:
            Dataset prepared for GRPO
        """
        # GRPO preparation is very similar to PPO preparation for now
        return self.prepare_for_ppo(
            dataset=dataset,
            problem_key=problem_key,
            reference_solution_key=reference_solution_key,
            apply_template=apply_template
        )
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create a DataLoader from a processed dataset.
        
        Args:
            dataset: Processed dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        def collate_fn(batch):
            # Convert list of dicts to dict of lists
            batch_dict = {key: [example[key] for example in batch] for key in batch[0].keys()}
            
            # Convert tensors to batched tensors
            for key in ["input_ids", "attention_mask", "labels"]:
                if key in batch_dict and isinstance(batch_dict[key][0], torch.Tensor):
                    batch_dict[key] = torch.stack(batch_dict[key])
                    
            return batch_dict
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> DatasetDict:
        """
        Split a dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            
        Returns:
            DatasetDict with 'train', 'validation', and 'test' splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        dataset = dataset.shuffle(seed=self.seed)
        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
    def save_dataset_splits(self, dataset_dict: DatasetDict, output_dir: str):
        """
        Save dataset splits to disk.
        
        Args:
            dataset_dict: DatasetDict with dataset splits
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, dataset in dataset_dict.items():
            output_path = os.path.join(output_dir, f"{split_name}.jsonl")
            logger.info(f"Saving {split_name} split with {len(dataset)} examples to {output_path}")
            
            with open(output_path, 'w') as f:
                for example in dataset:
                    # Convert to serializable format
                    serializable = {k: v for k, v in example.items() 
                                   if not isinstance(v, torch.Tensor)}
                    f.write(json.dumps(serializable) + '\n')
                    
        logger.info(f"All splits saved to {output_dir}")
        
    @staticmethod
    def load_prompt_templates(template_file: str) -> Dict[str, str]:
        """
        Load prompt templates from a JSON file.
        
        Args:
            template_file: Path to the template file
            
        Returns:
            Dictionary of prompt templates
        """
        with open(template_file, 'r') as f:
            templates = json.load(f)
        return templates 