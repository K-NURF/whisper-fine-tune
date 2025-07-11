#!/usr/bin/env python3
"""
Production-ready Whisper Fine-tuning Pipeline
Follows MLOps best practices with proper error handling and resource management.
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# ML/DL imports
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

# Handle different transformers versions for imports
try:
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
except ImportError:
    try:
        from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
        from transformers.trainer_seq2seq import Seq2SeqTrainer
    except ImportError:
        # For newer versions, these are in different locations
        from transformers import TrainingArguments as Seq2SeqTrainingArguments
        from transformers import Trainer as Seq2SeqTrainer

# Import callbacks separately with fallbacks
try:
    from transformers import TrainerCallback, EarlyStoppingCallback
except ImportError:
    try:
        from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
    except ImportError:
        # Minimal fallback implementations
        class TrainerCallback:
            def on_train_begin(self, args, state, control, **kwargs): pass
            def on_log(self, args, state, control, logs=None, **kwargs): pass
            def on_evaluate(self, args, state, control, metrics=None, **kwargs): pass
            def on_train_end(self, args, state, control, **kwargs): pass
        
        class EarlyStoppingCallback(TrainerCallback):
            def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0):
                self.early_stopping_patience = early_stopping_patience
                self.early_stopping_threshold = early_stopping_threshold
from datasets import Dataset, DatasetDict
import evaluate

# MLflow imports
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class WhisperTrainingConfig:
    """Configuration class for Whisper training pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Data paths
        self.labelstudio_export_path = "label-studio-export/data-v1/manifest.jsonl"
        self.audio_base_path = "label-studio-export/data-v1"
        
        # Model configuration
        self.model_name = "openai/whisper-small"
        self.target_language = "sw"  # Swahili
        self.task = "transcribe"
        
        # Training configuration
        self.learning_rate = 1e-5
        self.per_device_train_batch_size = 2
        self.per_device_eval_batch_size = 2
        self.gradient_accumulation_steps = 2
        self.max_steps = 1000
        self.warmup_steps = 500
        self.eval_steps = 100
        self.save_steps = 100
        self.logging_steps = 25
        
        # Memory optimization - FIXED FP16 ISSUES
        self.fp16 = False  # Disable FP16 to avoid gradient unscaling issues
        self.bf16 = True   # Use BF16 instead if available
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = {"use_reentrant": False}  # Fixed!
        self.dataloader_pin_memory = False
        self.dataloader_num_workers = 0
        
        # Evaluation
        self.test_size = 0.2
        self.max_eval_samples = 10
        
        # MLflow
        self.mlflow_tracking_uri = "http://localhost:5000"
        self.mlflow_experiment_name = "whisper-production-training"
        
        # Output
        self.output_dir = "./whisper-finetuned-production"
        self.registered_model_name = "whisper-finetuned-swahili"
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for MLflow logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models with proper padding"""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Cut bos token if appended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class MLflowIntegrationCallback(TrainerCallback):
    """Enhanced MLflow integration with proper error handling"""
    
    def __init__(self, config: WhisperTrainingConfig, baseline_wer: Optional[float] = None):
        self.config = config
        self.baseline_wer = baseline_wer
        self.run_id = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize MLflow run"""
        try:
            # End any existing runs
            if mlflow.active_run():
                mlflow.end_run()
            
            # Start new run
            run = mlflow.start_run(run_name=f"whisper-training-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
            self.run_id = run.info.run_id
            
            # Log configuration
            mlflow.log_params(self.config.to_dict())
            
            # Log training arguments
            training_params = {
                "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                "total_train_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, args.world_size),
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
            }
            mlflow.log_params(training_params)
            
            if self.baseline_wer is not None:
                mlflow.log_metric("baseline_wer", self.baseline_wer * 100)
            
            logger.info(f"MLflow run started: {self.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if logs and mlflow.active_run():
            try:
                step = state.global_step
                for key, value in logs.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(key, value, step=step)
                
                # Log epoch
                mlflow.log_metric("epoch", state.epoch, step=step)
                
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if metrics and mlflow.active_run():
            try:
                step = state.global_step
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(key, value, step=step)
                        
                logger.info(f"Evaluation at step {step}: {metrics}")
                
            except Exception as e:
                logger.warning(f"Failed to log evaluation metrics: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finalize MLflow run"""
        logger.info("Training completed. MLflow run ready for model registration.")


class WhisperDataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: WhisperTrainingConfig):
        self.config = config
        
    def load_labelstudio_export(self) -> List[Dict]:
        """Load and validate Label Studio export"""
        logger.info(f"Loading Label Studio export from {self.config.labelstudio_export_path}")
        
        processed_data = []
        with open(self.config.labelstudio_export_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    audio_path = item.get('audio_filepath')
                    transcription = item.get('text')
                    duration = item.get('duration')
                    
                    if audio_path and transcription:
                        transcription = transcription.strip()
                        
                        if 3 <= len(transcription) <= 1000:
                            processed_data.append({
                                'audio_path': audio_path,
                                'transcription': transcription,
                                'duration': duration
                            })
                            
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(processed_data)} valid samples")
        return processed_data
    
    def load_and_validate_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], Optional[int], str]:
        """Load and validate audio file"""
        try:
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.config.audio_base_path, audio_path)
            
            if not os.path.exists(audio_path):
                return None, None, f"File not found: {audio_path}"
            
            if os.path.getsize(audio_path) == 0:
                return None, None, f"Empty file: {audio_path}"
            
            audio, sr = sf.read(audio_path)
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if sr != target_sr:
                return None, None, f"Unexpected sample rate {sr}Hz (expected {target_sr}Hz)"
            
            if audio is None or len(audio) == 0:
                return None, None, f"Empty audio data"
            
            duration = len(audio) / sr
            if duration < 0.1 or duration > 60:
                return None, None, f"Unexpected duration ({duration:.2f}s)"
            
            return audio, sr, "OK"
            
        except Exception as e:
            return None, None, f"Error loading {audio_path}: {str(e)}"
    
    def process_audio_samples(self, labelstudio_data: List[Dict]) -> List[Dict]:
        """Process and filter audio samples"""
        logger.info("Processing audio samples...")
        
        processed_samples = []
        failed_count = 0
        
        for i, sample in enumerate(labelstudio_data):
            audio, sr, status = self.load_and_validate_audio(sample['audio_path'])
            
            if audio is not None:
                processed_samples.append({
                    'audio': audio,
                    'transcription': sample['transcription'],
                    'audio_path': sample['audio_path'],
                    'duration': len(audio) / sr
                })
            else:
                failed_count += 1
                if i < 5:  # Log first few failures
                    logger.warning(f"Failed to load {sample['audio_path']}: {status}")
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(labelstudio_data)} samples, {len(processed_samples)} valid")
        
        logger.info(f"Final dataset: {len(processed_samples)} valid samples, {failed_count} failed")
        
        if processed_samples:
            durations = [s['duration'] for s in processed_samples]
            logger.info(f"Audio stats - Mean: {np.mean(durations):.2f}s, "
                       f"Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s, "
                       f"Total: {np.sum(durations)/60:.1f} minutes")
        
        return processed_samples


class WhisperTrainingPipeline:
    """Main training pipeline"""
    
    def __init__(self, config: WhisperTrainingConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.trainer = None
        self.wer_metric = evaluate.load("wer")
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Clear GPU memory
        self._clear_gpu_memory()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        try:
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            logger.info(f"MLflow experiment set: {self.config.mlflow_experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup issue: {e}")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory and clean up"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")
    
    def load_model_and_processor(self):
        """Load Whisper model and processor with proper error handling"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
            
            # Load model with appropriate dtype
            model_kwargs = {}
            if torch.cuda.is_available():
                model_kwargs['torch_dtype'] = torch.float16 if self.config.fp16 else torch.float32
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name, 
                **model_kwargs
            )
            
            # Configure model for training
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            self.model.config.use_cache = False  # Important for gradient checkpointing
            
            # Move to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_dataset(self, samples: List[Dict]) -> DatasetDict:
        """Convert samples to HuggingFace dataset format"""
        logger.info("Preparing dataset...")
        
        def process_sample(sample):
            # Process audio to log-mel spectrogram
            input_features = self.processor.feature_extractor(
                sample['audio'], 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features[0]
            
            # Tokenize transcription
            labels = self.processor.tokenizer(
                sample['transcription'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=448
            ).input_ids[0]
            
            return {
                "input_features": input_features,
                "labels": labels
            }
        
        # Process all samples
        processed = []
        for sample in samples:
            try:
                processed_sample = process_sample(sample)
                processed.append(processed_sample)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
        
        if not processed:
            raise ValueError("No samples could be processed successfully")
        
        # Create dataset
        dataset = Dataset.from_list(processed)
        
        # Train/test split
        dataset = dataset.train_test_split(test_size=self.config.test_size, seed=42)
        
        logger.info(f"Dataset prepared - Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Compute WER metric during training"""
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Handle nested predictions (common in Seq2Seq models)
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]  # Take the first element if it's a tuple
        
        # Ensure pred_ids is 2D (batch_size, sequence_length)
        if len(pred_ids.shape) > 2:
            pred_ids = pred_ids.reshape(-1, pred_ids.shape[-1])
        
        # Replace -100 with pad token id in labels
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        try:
            # Decode predictions and labels with error handling
            pred_str = []
            label_str = []
            
            for i in range(len(pred_ids)):
                try:
                    # Handle individual prediction
                    pred_tokens = pred_ids[i]
                    if isinstance(pred_tokens, list):
                        pred_tokens = pred_tokens[0] if len(pred_tokens) > 0 and isinstance(pred_tokens[0], list) else pred_tokens
                    
                    # Ensure pred_tokens is a 1D array/list of integers
                    if hasattr(pred_tokens, 'flatten'):
                        pred_tokens = pred_tokens.flatten()
                    
                    # Convert to list and filter out non-integer values
                    pred_tokens = [int(token) for token in pred_tokens if isinstance(token, (int, float)) and not isinstance(token, bool)]
                    
                    # Decode
                    pred_text = self.processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    pred_str.append(pred_text)
                    
                except Exception as e:
                    logger.warning(f"Failed to decode prediction {i}: {e}")
                    pred_str.append("")  # Empty string as fallback
                
                try:
                    # Handle labels
                    label_tokens = label_ids[i]
                    label_text = self.processor.tokenizer.decode(label_tokens, skip_special_tokens=True)
                    label_str.append(label_text)
                    
                except Exception as e:
                    logger.warning(f"Failed to decode label {i}: {e}")
                    label_str.append("")  # Empty string as fallback

            # Compute WER with error handling
            if pred_str and label_str:
                wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
            else:
                logger.warning("No valid predictions/labels for WER computation")
                wer = 1.0  # Return high WER if computation fails

        except Exception as e:
            logger.warning(f"WER computation failed: {e}")
            wer = 1.0  # Return high WER if computation fails

        return {"wer": wer}
    
    def evaluate_baseline(self, test_samples: List[Dict], max_samples: int = 10) -> Optional[float]:
        """Evaluate baseline model performance"""
        logger.info("Evaluating baseline model...")
        
        predictions = []
        references = []
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for i, sample in enumerate(test_samples[:max_samples]):
                try:
                    # Process audio
                    input_features = self.processor(
                        sample['audio'], 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_features.to(device)
                    
                    # Generate transcription with safer parameters
                    try:
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                            language=self.config.target_language, 
                            task=self.config.task
                        )
                    except Exception:
                        # Fallback if get_decoder_prompt_ids fails
                        forced_decoder_ids = None
                    
                    generation_kwargs = {
                        "max_length": 225,
                        "num_beams": 1,  # Use greedy decoding for baseline
                        "do_sample": False
                    }
                    
                    if forced_decoder_ids is not None:
                        generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
                    
                    predicted_ids = self.model.generate(
                        input_features,
                        **generation_kwargs
                    )[0]
                    
                    prediction = self.processor.decode(predicted_ids, skip_special_tokens=True)
                    
                    predictions.append(prediction)
                    references.append(sample['transcription'])
                    
                except Exception as e:
                    logger.warning(f"Baseline evaluation failed for sample {i}: {e}")
                    continue
        
        if predictions:
            try:
                baseline_wer = self.wer_metric.compute(predictions=predictions, references=references)
                logger.info(f"Baseline WER: {baseline_wer:.4f}")
                
                # Log sample predictions
                for i in range(min(3, len(predictions))):
                    logger.info(f"Sample {i+1}:")
                    logger.info(f"  Reference: {references[i][:100]}...")
                    logger.info(f"  Prediction: {predictions[i][:100]}...")
                
                return baseline_wer
            except Exception as e:
                logger.warning(f"WER computation failed: {e}")
                return None
        else:
            logger.warning("No successful baseline predictions")
            return None
    
    def setup_trainer(self, dataset: DatasetDict, baseline_wer: Optional[float] = None):
        """Setup the Seq2SeqTrainer"""
        logger.info("Setting up trainer...")
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            
            # Batch configuration
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Training configuration
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            
            # Memory optimization - FIXED FP16 ISSUES
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=False,  # Disable FP16 to avoid gradient unscaling errors
            bf16=True,   # Use BF16 instead (better compatibility)
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            generation_max_length=225,
            
            # Saving and logging
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            
            # Disable automatic MLflow reporting (we use custom callback)
            report_to=[],
            
            # Memory optimization
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # Remove problematic settings for compatibility
            remove_unused_columns=False,  # Important for speech tasks
            
            # Additional stability settings
            max_grad_norm=1.0,  # Gradient clipping
            optim="adamw_torch",  # Use torch AdamW optimizer
        )
        
        # Add gradient checkpointing kwargs only if supported
        try:
            if hasattr(training_args, 'gradient_checkpointing_kwargs'):
                training_args.gradient_checkpointing_kwargs = self.config.gradient_checkpointing_kwargs
        except Exception:
            pass  # Ignore if not supported in this version
        
        # Auto-detect and set appropriate precision
        if torch.cuda.is_available():
            # Check if BF16 is supported (Ampere GPUs and newer)
            if torch.cuda.get_device_capability()[0] >= 8:
                training_args.bf16 = True
                training_args.fp16 = False
                logger.info("Using BF16 precision (Ampere GPU detected)")
            else:
                # Fallback to FP32 for older GPUs to avoid gradient unscaling issues
                training_args.bf16 = False
                training_args.fp16 = False
                logger.info("Using FP32 precision (older GPU or FP16 compatibility issues)")
        else:
            training_args.bf16 = False
            training_args.fp16 = False
            logger.info("Using FP32 precision (CPU training)")
        
        # Initialize trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset["test"],
            "data_collator": data_collator,
            "compute_metrics": self.compute_metrics,
        }
        
        # Add processing_class only if supported
        try:
            trainer_kwargs["processing_class"] = self.processor
        except Exception:
            # Fallback for older versions that use tokenizer
            try:
                trainer_kwargs["tokenizer"] = self.processor.tokenizer
            except Exception:
                pass  # Some versions don't need this
        
        self.trainer = Seq2SeqTrainer(**trainer_kwargs)
        
        # Add callbacks
        mlflow_callback = MLflowIntegrationCallback(self.config, baseline_wer)
        self.trainer.add_callback(mlflow_callback)
        
        # Add early stopping (only if callback class is available)
        try:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            self.trainer.add_callback(early_stopping)
        except Exception:
            logger.info("Early stopping callback not available in this version")
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Execute training with proper error handling"""
        logger.info("Starting training...")
        
        try:
            # Clear memory before training
            self._clear_gpu_memory()
            
            # Train
            train_result = self.trainer.train()
            
            logger.info("Training completed successfully")
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clean up
            self._clear_gpu_memory()
    
    def evaluate_and_save_model(self) -> Dict:
        """Final evaluation and model saving"""
        logger.info("Final evaluation...")
        
        try:
            # Evaluate
            eval_results = self.trainer.evaluate()
            final_wer = eval_results.get('eval_wer', None)
            
            logger.info(f"Final evaluation results: {eval_results}")
            
            # Save model locally
            self.trainer.save_model()
            logger.info(f"Model saved to {self.config.output_dir}")
            
            # Log to MLflow if active run
            if mlflow.active_run():
                try:
                    # Log final metrics
                    for key, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"final_{key}", value)
                    
                    # Save model to MLflow
                    model_to_save = self.trainer.model
                    if hasattr(self.trainer, 'accelerator'):
                        try:
                            model_to_save = self.trainer.accelerator.unwrap_model(self.trainer.model)
                        except Exception:
                            pass  # Use original model if unwrapping fails
                    
                    # Simple model logging without signature for compatibility
                    try:
                        mlflow.pytorch.log_model(
                            pytorch_model=model_to_save,
                            artifact_path="model",
                            registered_model_name=self.config.registered_model_name
                        )
                        logger.info("Model logged to MLflow successfully")
                    except Exception as e:
                        logger.warning(f"MLflow model logging failed: {e}")
                        # Try alternative approach
                        try:
                            import tempfile
                            import os
                            
                            with tempfile.TemporaryDirectory() as temp_dir:
                                model_path = os.path.join(temp_dir, "model")
                                model_to_save.save_pretrained(model_path)
                                self.processor.save_pretrained(model_path)
                                mlflow.log_artifacts(model_path, artifact_path="model_files")
                                logger.info("Model files logged to MLflow as artifacts")
                        except Exception as e2:
                            logger.warning(f"Alternative model logging also failed: {e2}")
                    
                except Exception as e:
                    logger.warning(f"MLflow model logging failed: {e}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            raise
        finally:
            # End MLflow run
            try:
                if mlflow.active_run():
                    mlflow.end_run()
            except:
                pass
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting complete Whisper training pipeline...")
        
        try:
            # 1. Load data
            data_loader = WhisperDataLoader(self.config)
            labelstudio_data = data_loader.load_labelstudio_export()
            processed_samples = data_loader.process_audio_samples(labelstudio_data)
            
            if len(processed_samples) == 0:
                raise ValueError("No valid audio samples found")
            
            # 2. Load model
            self.load_model_and_processor()
            
            # 3. Prepare dataset
            dataset = self.prepare_dataset(processed_samples)
            
            # 4. Evaluate baseline
            baseline_wer = self.evaluate_baseline(processed_samples[-20:], self.config.max_eval_samples)
            
            # 5. Setup trainer
            self.setup_trainer(dataset, baseline_wer)
            
            # 6. Train
            train_result = self.train()
            
            # 7. Final evaluation
            eval_results = self.evaluate_and_save_model()
            
            # 8. Summary
            logger.info("="*60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            if baseline_wer:
                logger.info(f"Baseline WER: {baseline_wer*100:.2f}%")
            if 'eval_wer' in eval_results:
                final_wer = eval_results['eval_wer']
                logger.info(f"Final WER: {final_wer:.2f}%")
                if baseline_wer:
                    improvement = ((baseline_wer*100 - final_wer) / (baseline_wer*100)) * 100
                    logger.info(f"Improvement: {improvement:.2f}%")
            
            logger.info(f"Model saved to: {self.config.output_dir}")
            logger.info(f"MLflow tracking: {self.config.mlflow_tracking_uri}")
            
            return {
                'train_result': train_result,
                'eval_results': eval_results,
                'baseline_wer': baseline_wer
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self._clear_gpu_memory()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Whisper Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--labelstudio-export", type=str, help="Path to Label Studio export")
    parser.add_argument("--audio-base-path", type=str, help="Base path for audio files")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small", help="Whisper model name")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./whisper-finetuned-production", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = WhisperTrainingConfig(args.config)
    
    # Override with command line arguments
    if args.labelstudio_export:
        config.labelstudio_export_path = args.labelstudio_export
    if args.audio_base_path:
        config.audio_base_path = args.audio_base_path
    if args.model_name:
        config.model_name = args.model_name
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run pipeline
    pipeline = WhisperTrainingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    logger.info("Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    main()