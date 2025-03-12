"""
Evaluation framework for assessing language model performance on mathematical tasks.
"""

import os
import json
import yaml
import logging
import time
import random
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..models.finemathllama import FineMathLlamaModel
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

class MathEvaluator:
    """
    Evaluation framework for assessing language model performance on mathematical reasoning tasks.
    Compares baseline, PPO, and GRPO models on various mathematical benchmarks.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            config_path: Path to evaluation configuration YAML file
            config: Configuration dictionary (used if config_path is None)
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        # Load configuration
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}
            
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Initialize empty model containers
        self.models = {}
        self.tokenizers = {}
        
        # Initialize empty dataset containers
        self.eval_datasets = {}
        
        # Configure logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str = "baseline",
        use_gguf: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer from a specified path.
        
        Args:
            model_name: Name to use for this model in the evaluator
            model_path: Path to the model or HF model ID
            model_type: Type of model (baseline, rl, etc.)
            use_gguf: Whether to use GGUF model loading via ctransformers
            
        Returns:
            Tuple containing (model, tokenizer)
        """
        logger.info(f"Loading {model_type} model from {model_path}")
        
        try:
            if use_gguf:
                # Load GGUF model using ctransformers
                logger.info("Using ctransformers to load GGUF model")
                from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
                from ctransformers import AutoTokenizer as CTAutoTokenizer
                
                # Load the model and tokenizer
                model = CTAutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type="llama",  # Assuming it's a Llama model
                    gpu_layers=50,       # Use as many layers on GPU as possible
                    threads=8            # Number of CPU threads to use
                )
                
                # For GGUF models, we might need to use the original model's tokenizer
                tokenizer_repo = model_path if "/" in model_path else "HuggingFaceTB/FineMath-Llama-3B"
                tokenizer = CTAutoTokenizer.from_pretrained(tokenizer_repo)
                
                # Set pad token if not defined
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            else:
                # Simple direct loading approach similar to HuggingFace examples
                logger.info("Loading tokenizer with trust_remote_code=True")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True,
                    use_fast=False  # Using slow tokenizer for better compatibility
                )
                
                # Ensure the tokenizer has pad_token set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Determine if we should use 4-bit quantization
                use_4bit = self.config.get("use_4bit", True)
                
                logger.info(f"Loading model with use_4bit={use_4bit}, trust_remote_code=True")
                
                try:
                    if use_4bit:
                        # Load in 4-bit to save memory
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            ignore_mismatched_sizes=True,
                            low_cpu_mem_usage=True
                        )
                    else:
                        # Load without quantization
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            ignore_mismatched_sizes=True,
                            low_cpu_mem_usage=True
                        )
                except ValueError as e:
                    if "rope_scaling" in str(e):
                        logger.warning("Encountered rope_scaling issue. Attempting to load with different config.")
                        
                        # Try alternative loading approach
                        from transformers import AutoConfig
                        
                        # Get configuration but don't validate
                        config = AutoConfig.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                        )
                        
                        # If it has rope_scaling, modify it to be compatible
                        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                            logger.info(f"Original rope_scaling: {config.rope_scaling}")
                            # Simplify the rope_scaling config
                            if "factor" in config.rope_scaling:
                                config.rope_scaling = {
                                    "type": "linear",
                                    "factor": float(config.rope_scaling["factor"])
                                }
                                logger.info(f"Modified rope_scaling: {config.rope_scaling}")
                        
                        # Try again with our modified config
                        if use_4bit:
                            from transformers import BitsAndBytesConfig
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                config=config,
                                quantization_config=quantization_config,
                                torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                                device_map="auto",
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True,
                                low_cpu_mem_usage=True
                            )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                config=config,
                                torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                                device_map="auto",
                                trust_remote_code=True,
                                ignore_mismatched_sizes=True,
                                low_cpu_mem_usage=True
                            )
                    else:
                        # Re-raise if not a rope_scaling issue
                        raise
                
            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded model {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def load_evaluation_datasets(self) -> Dict[str, Dataset]:
        """
        Load evaluation datasets based on configuration.
        
        Returns:
            Dictionary of dataset name to Dataset object
        """
        logger.info("Loading evaluation datasets")
        
        eval_datasets_config = self.config.get("eval_datasets", {})
        
        for dataset_name, dataset_config in eval_datasets_config.items():
            logger.info(f"Loading dataset: {dataset_name}")
            
            path = dataset_config.get("path")
            split = dataset_config.get("split")
            subset = dataset_config.get("subset")
            max_samples = dataset_config.get("max_samples")
            
            try:
                # Check if path is a local file
                if os.path.exists(path):
                    # Load from local file (JSON or JSONL)
                    dataset = load_dataset('json', data_files=path, split='train')
                else:
                    # Load from HuggingFace datasets
                    dataset = load_dataset(path, subset, split=split)
                    
                # Limit number of samples if specified
                if max_samples and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                    
                self.eval_datasets[dataset_name] = dataset
                logger.info(f"Loaded {len(dataset)} examples for {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                
        return self.eval_datasets
        
    def get_prompt_template(self, dataset_name: str) -> str:
        """
        Get the appropriate prompt template for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Prompt template string
        """
        # Get prompt templates from config
        prompt_templates = self.config.get("prompt_templates", {})
        
        # Try to get dataset-specific template
        if dataset_name in prompt_templates:
            return prompt_templates[dataset_name]
            
        # Fall back to default template
        if "default" in prompt_templates:
            return prompt_templates["default"]
            
        # Return a basic default template if nothing is configured
        return "Solve the following math problem step by step:\n{problem}\n\nSolution:"
    
    def generate_responses(
        self,
        model_name: str,
        dataset: Dataset,
        problem_key: str = "question",
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> List[str]:
        """
        Generate responses for all problems in a dataset.
        
        Args:
            model_name: Name of the model to use
            dataset: Dataset containing problems
            problem_key: Key for the problem text in the dataset
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            List of generated responses
        """
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        
        if model is None or tokenizer is None:
            raise ValueError(f"Model {model_name} not loaded")
            
        # Get prompt template for the dataset name
        # Extract dataset name from keys
        dataset_name = next((k for k, v in self.eval_datasets.items() if v == dataset), "default")
        prompt_template = self.get_prompt_template(dataset_name)
        
        # Prepare prompts
        problems = dataset[problem_key]
        prompts = [prompt_template.format(problem=problem) for problem in problems]
        
        logger.info(f"Generating {len(prompts)} responses with {model_name}")
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": self.config.get("generation", {}).get("top_p", 0.95),
            "top_k": self.config.get("generation", {}).get("top_k", 50),
            "num_beams": self.config.get("generation", {}).get("num_beams", 1),
            "repetition_penalty": self.config.get("generation", {}).get("repetition_penalty", 1.1),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "early_stopping": self.config.get("generation", {}).get("early_stopping", True),
        }
        
        # Generate responses in batches
        responses = []
        
        # Put model in evaluation mode
        model.eval()
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with {model_name}"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize inputs
            tokenized_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("generation", {}).get("max_length", 1024)
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    **gen_kwargs
                )
                
            # Extract responses
            for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
                # Decode and extract only the generated part
                encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_len = len(encoded_prompt)
                
                generated_tokens = output[prompt_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                responses.append(generated_text)
                
        return responses
    
    def evaluate_model(
        self,
        model_name: str,
        dataset_name: str,
        problem_key: str = "question",
        reference_key: str = "answer",
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model_name: Name of the model to evaluate
            dataset_name: Name of the dataset to evaluate on
            problem_key: Key for the problem text in the dataset
            reference_key: Key for the reference answer in the dataset
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Dictionary of evaluation metrics
        """
        dataset = self.eval_datasets.get(dataset_name)
        
        if dataset is None:
            raise ValueError(f"Dataset {dataset_name} not loaded")
            
        # Generate responses
        start_time = time.time()
        responses = self.generate_responses(
            model_name=model_name,
            dataset=dataset,
            problem_key=problem_key,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
        generation_time = time.time() - start_time
        
        # Get reference answers if available
        references = dataset[reference_key] if reference_key in dataset.features else None
        
        # Compute metrics
        metrics = compute_metrics(
            predictions=responses,
            references=references,
            problems=dataset[problem_key],
            metrics_list=self.config.get("metrics", ["exact_match", "f1_score"])
        )
        
        # Add timing information
        metrics["generation_time"] = generation_time
        metrics["generation_time_per_example"] = generation_time / len(dataset)
        
        # Store predictions and metrics
        output_dir = self.config.get("output_dir", "results")
        if output_dir:
            os.makedirs(output_dir.format(model_name=model_name), exist_ok=True)
            
            # Save predictions
            if self.config.get("analysis", {}).get("save_predictions", True):
                predictions_file = os.path.join(
                    output_dir.format(model_name=model_name),
                    f"{dataset_name}_predictions.json"
                )
                
                with open(predictions_file, 'w') as f:
                    json.dump(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "problems": dataset[problem_key],
                            "predictions": responses,
                            "references": references if references else [],
                            "metrics": metrics
                        },
                        f,
                        indent=2
                    )
                
            # Save metrics separately
            metrics_file = os.path.join(
                output_dir.format(model_name=model_name),
                f"{dataset_name}_metrics.json"
            )
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        return metrics
    
    def compare_models(
        self,
        model_names: List[str],
        dataset_name: str,
        problem_key: str = "question",
        reference_key: str = "answer",
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on a dataset.
        
        Args:
            model_names: List of model names to compare
            dataset_name: Name of the dataset to evaluate on
            problem_key: Key for the problem text in the dataset
            reference_key: Key for the reference answer in the dataset
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Dictionary mapping model names to metrics
        """
        results = {}
        
        for model_name in model_names:
            logger.info(f"Evaluating model: {model_name} on {dataset_name}")
            
            try:
                metrics = self.evaluate_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    problem_key=problem_key,
                    reference_key=reference_key,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )
                
                results[model_name] = metrics
                
                logger.info(f"Evaluation results for {model_name} on {dataset_name}:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: {metric_value}")
                        
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                
        # Save comparison results
        output_dir = self.config.get("output_dir", "results")
        if output_dir:
            os.makedirs(output_dir.format(model_name="comparison"), exist_ok=True)
            
            comparison_file = os.path.join(
                output_dir.format(model_name="comparison"),
                f"{dataset_name}_model_comparison.json"
            )
            
            with open(comparison_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Generate comparison plots if configured
            if self.config.get("analysis", {}).get("generate_plots", True):
                self._generate_comparison_plots(results, dataset_name)
                
        return results
    
    def evaluate_all_models_all_datasets(
        self,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluate all loaded models on all loaded datasets.
        
        Args:
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Nested dictionary: model_name -> dataset_name -> metrics
        """
        all_results = {}
        
        for model_name in self.models:
            all_results[model_name] = {}
            
            for dataset_name in self.eval_datasets:
                logger.info(f"Evaluating {model_name} on {dataset_name}")
                
                try:
                    metrics = self.evaluate_model(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample
                    )
                    
                    all_results[model_name][dataset_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    all_results[model_name][dataset_name] = {"error": str(e)}
                    
        # Save all results
        output_dir = self.config.get("output_dir", "results")
        if output_dir:
            os.makedirs(output_dir.format(model_name="all"), exist_ok=True)
            
            all_results_file = os.path.join(
                output_dir.format(model_name="all"),
                "all_evaluation_results.json"
            )
            
            with open(all_results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
        return all_results
    
    def _generate_comparison_plots(
        self,
        results: Dict[str, Dict[str, Any]],
        dataset_name: str
    ):
        """
        Generate comparison plots for visualization.
        
        Args:
            results: Dictionary mapping model names to metrics
            dataset_name: Name of the dataset
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the plotting style
            sns.set(style="whitegrid")
            plt.rcParams.update({"font.size": 12})
            
            # Extract metrics that are numeric for plotting
            metrics_to_plot = {}
            for model_name, metrics in results.items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not metric_name.startswith("generation_time"):
                        if metric_name not in metrics_to_plot:
                            metrics_to_plot[metric_name] = {}
                        metrics_to_plot[metric_name][model_name] = metric_value
                            
            # Plot each metric
            for metric_name, model_values in metrics_to_plot.items():
                plt.figure(figsize=(10, 6))
                
                # Create bar plot
                sns.barplot(
                    x=list(model_values.keys()),
                    y=list(model_values.values()),
                    palette="viridis"
                )
                
                plt.title(f"{metric_name} Comparison on {dataset_name}")
                plt.xlabel("Model")
                plt.ylabel(metric_name)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                output_dir = self.config.get("output_dir", "results")
                if output_dir:
                    plot_dir = os.path.join(output_dir.format(model_name="comparison"), "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    plot_file = os.path.join(plot_dir, f"{dataset_name}_{metric_name}_comparison.png")
                    plt.savefig(plot_file, dpi=300)
                    
                plt.close()
                
            # Create a summary plot with all main metrics
            main_metrics = ["exact_match", "f1_score", "mathematical_accuracy"]
            main_metrics = [m for m in main_metrics if m in metrics_to_plot]
            
            if main_metrics:
                plt.figure(figsize=(12, 8))
                
                # Prepare data for grouped bar plot
                models = list(results.keys())
                x = np.arange(len(models))
                width = 0.8 / len(main_metrics)
                
                for i, metric in enumerate(main_metrics):
                    values = [metrics_to_plot[metric].get(model, 0) for model in models]
                    plt.bar(x + i * width - 0.4 + width/2, values, width, label=metric)
                    
                plt.xlabel("Model")
                plt.ylabel("Score")
                plt.title(f"Performance Comparison on {dataset_name}")
                plt.xticks(x, models, rotation=45)
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                output_dir = self.config.get("output_dir", "results")
                if output_dir:
                    plot_dir = os.path.join(output_dir.format(model_name="comparison"), "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    plot_file = os.path.join(plot_dir, f"{dataset_name}_summary_comparison.png")
                    plt.savefig(plot_file, dpi=300)
                    
                plt.close()
                
        except ImportError:
            logger.warning("Matplotlib or seaborn not available. Skipping plot generation.")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            
    def generate_qualitative_examples(
        self,
        model_names: List[str],
        dataset_name: str,
        num_examples: int = 5,
        problem_key: str = "question",
        reference_key: str = "answer",
        max_new_tokens: int = 512
    ):
        """
        Generate qualitative examples for side-by-side comparison.
        
        Args:
            model_names: List of model names to compare
            dataset_name: Name of the dataset to evaluate on
            num_examples: Number of examples to generate
            problem_key: Key for the problem text in the dataset
            reference_key: Key for the reference answer in the dataset
            max_new_tokens: Maximum number of new tokens to generate
        """
        dataset = self.eval_datasets.get(dataset_name)
        
        if dataset is None:
            raise ValueError(f"Dataset {dataset_name} not loaded")
            
        # Randomly select examples
        num_examples = min(num_examples, len(dataset))
        example_indices = random.sample(range(len(dataset)), num_examples)
        
        examples = []
        
        for idx in example_indices:
            problem = dataset[idx][problem_key]
            reference = dataset[idx][reference_key] if reference_key in dataset[idx] else None
            
            # Get prompt for the problem
            prompt_template = self.get_prompt_template(dataset_name)
            prompt = prompt_template.format(problem=problem)
            
            # Generate response from each model
            model_responses = {}
            
            for model_name in model_names:
                model = self.models.get(model_name)
                tokenizer = self.tokenizers.get(model_name)
                
                if model is None or tokenizer is None:
                    logger.warning(f"Model {model_name} not loaded. Skipping.")
                    continue
                    
                # Tokenize input
                tokenized_input = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.config.get("generation", {}).get("max_length", 1024)
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    output = model.generate(
                        input_ids=tokenized_input.input_ids,
                        attention_mask=tokenized_input.attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=False,
                        top_p=0.95,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )[0]
                    
                # Decode and extract only the generated part
                encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_len = len(encoded_prompt)
                
                generated_tokens = output[prompt_len:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                model_responses[model_name] = response
                
            # Create example
            example = {
                "problem": problem,
                "reference": reference,
                "responses": model_responses
            }
            
            examples.append(example)
            
        # Save examples
        output_dir = self.config.get("output_dir", "results")
        if output_dir:
            os.makedirs(output_dir.format(model_name="comparison"), exist_ok=True)
            
            examples_file = os.path.join(
                output_dir.format(model_name="comparison"),
                f"{dataset_name}_qualitative_examples.json"
            )
            
            with open(examples_file, 'w') as f:
                json.dump(examples, f, indent=2)
                
        logger.info(f"Generated {len(examples)} qualitative examples for comparison")
        return examples 