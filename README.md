# Whisper Fine-tuning Production Pipeline

A production-ready MLOps pipeline for fine-tuning OpenAI's Whisper models on custom datasets with comprehensive experiment tracking and monitoring.

## 🚀 Features

- **Production-Ready Architecture**: Structured codebase following MLOps best practices
- **Comprehensive MLflow Integration**: Experiment tracking, model registry, and metrics visualization
- **Memory Optimization**: Proper GPU memory management and gradient checkpointing
- **Error Handling**: Robust error handling and recovery mechanisms
- **Flexible Configuration**: JSON-based configuration management
- **Automated Setup**: One-command environment setup and training execution

## 🔧 Setup

### 1. Environment Setup

```bash
# Clone or download the training files
mkdir whisper-training && cd whisper-training

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your Label Studio export is in the correct format:

```
project-directory/
├── label-studio-export/
│   └── data-v1/
│       ├── manifest.jsonl          # Label Studio export
│       └── audio/                  # Audio files directory
│           ├── chunk_xxx.wav
│           ├── chunk_yyy.wav
│           └── ...
├── config/
│   └── training_config.json        # Training configuration
├── whisper_training_pipeline.py    # Main training script
└── setup_training.py              # Environment setup
```

**Expected Label Studio JSONL format:**
```json
{"audio_filepath": "audio/chunk_xxx.wav", "duration": 12.277875, "text": "transcription text"}
{"audio_filepath": "audio/chunk_yyy.wav", "duration": 11.961875, "text": "another transcription"}
```

### 3. Configuration

Edit `config/training_config.json` to match your setup:

```json
{
  "labelstudio_export_path": "label-studio-export/data-v1/manifest.jsonl",
  "audio_base_path": "label-studio-export/data-v1",
  "model_name": "openai/whisper-small",
  "max_steps": 1000,
  "learning_rate": 1e-5
}
```

## 🏃 Quick Start

### Option 1: Automated Setup and Training

```bash
# Make run script executable
chmod +x run_training.sh

# Run complete pipeline
./run_training.sh
```

### Option 2: Manual Setup

```bash
# 1. Setup environment and MLflow server
python3 setup_training.py

# 2. Run training
python3 whisper_training_pipeline.py --config config/training_config.json

# 3. Monitor at http://localhost:5000
```

### Option 3: Custom Configuration

```bash
python3 whisper_training_pipeline.py \
    --labelstudio-export path/to/your/manifest.jsonl \
    --audio-base-path path/to/audio/directory \
    --model-name openai/whisper-medium \
    --max-steps 2000 \
    --learning-rate 5e-6 \
    --output-dir ./my-custom-model
```

## 📊 Monitoring and Results

### MLflow Dashboard
- **URL**: http://localhost:5000
- **Features**: 
  - Real-time training metrics
  - Model comparison
  - Experiment tracking
  - Model registry

### Key Metrics Tracked
- Training/Validation Loss
- Word Error Rate (WER)
- Learning Rate Schedule
- GPU Memory Usage
- Training Speed

### Model Outputs
- **Local Save**: `./whisper-finetuned-production/`
- **MLflow Registry**: Models registered as `whisper-finetuned-swahili`
- **Artifacts**: Training logs, configuration, metrics

## 🛠 Key Improvements Over Notebook Version

### 1. **Fixed Runtime Error**
- ✅ Proper gradient checkpointing configuration (`use_reentrant=False`)
- ✅ Memory management and cleanup
- ✅ Model state isolation

### 2. **Production Architecture**
- ✅ Modular, reusable components
- ✅ Configuration management
- ✅ Comprehensive logging
- ✅ Error handling and recovery

### 3. **MLOps Best Practices**
- ✅ Experiment tracking with MLflow
- ✅ Model versioning and registry
- ✅ Reproducible training runs
- ✅ Automated baseline comparison

### 4. **Resource Optimization**
- ✅ GPU memory optimization
- ✅ Efficient data loading
- ✅ Gradient accumulation
- ✅ Mixed precision training

## 📁 Project Structure

```
whisper-training/
├── whisper_training_pipeline.py    # Main training pipeline
├── setup_training.py              # Environment setup script
├── run_training.sh                # Automated training script
├── requirements.txt               # Python dependencies
├── config/
│   └── training_config.json      # Training configuration
├── logs/                         # Training logs
├── mlruns/                       # MLflow artifacts
├── models/                       # Saved models
└── whisper-finetuned-production/ # Training output
```

## ⚙ Configuration Options

### Model Configuration
- `model_name`: Whisper model variant (`whisper-small`, `whisper-medium`, `whisper-large`)
- `target_language`: Target language code (e.g., `"sw"` for Swahili)
- `task`: Task type (`"transcribe"` or `"translate"`)

### Training Parameters
- `learning_rate`: Learning rate (default: `1e-5`)
- `max_steps`: Maximum training steps (default: `1000`)
- `per_device_train_batch_size`: Batch size per GPU (default: `2`)
- `gradient_accumulation_steps`: Gradient accumulation (default: `2`)

### Memory Optimization
- `fp16`: Mixed precision training (default: `true`)
- `gradient_checkpointing`: Memory-efficient training (default: `true`)
- `dataloader_num_workers`: Data loading workers (default: `0`)

## 🔍 Troubleshooting

### Common Issues

1. **"Trying to backward through the graph a second time"**
   - ✅ **Fixed**: Proper gradient checkpointing configuration
   - ✅ **Fixed**: Memory cleanup between phases

2. **Out of Memory Errors**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `fp16` training

3. **MLflow Connection Issues**
   - Run `python3 setup_training.py` to start MLflow server
   - Check if port 5000 is available
   - Verify `mlflow_tracking_uri` in config

4. **Data Loading Issues**
   - Verify Label Studio export format
   - Check audio file paths and accessibility
   - Ensure proper file permissions

### Performance Optimization

```bash
# For larger datasets
python3 whisper_training_pipeline.py \
    --model-name openai/whisper-medium \
    --max-steps 2000 \
    --learning-rate 5e-6

# For limited GPU memory
python3 whisper_training_pipeline.py \
    --config config/training_config.json \
    # Edit config to reduce batch_size and enable fp16
```

## 📈 Expected Results

### Training Progress
- **Initial WER**: ~200-300% (baseline Whisper on domain data)
- **Target WER**: <50% (significant improvement expected)
- **Training Time**: ~30-60 minutes for 1000 steps on RTX 3050

### Model Performance
- **Domain Adaptation**: Improved transcription for domain-specific audio
- **Language Handling**: Better handling of target language nuances
- **Quality Improvements**: Reduced word errors and better punctuation

## 🔄 Integration with Existing MLOps Stack

This pipeline integrates seamlessly with:
- **MLflow**: Experiment tracking and model registry
- **Docker**: Containerized training (add Dockerfile as needed)
- **CI/CD**: Automated retraining pipelines
- **Cloud Platforms**: Easy deployment to AWS/GCP/Azure

## 📚 Next Steps

1. **Scale Up**: Train with larger datasets and longer training
2. **Model Variants**: Experiment with different Whisper sizes
3. **Advanced Features**: Add data augmentation and cross-validation
4. **Production Deployment**: Set up model serving infrastructure
5. **Monitoring**: Implement production model monitoring

## 🤝 Contributing

This pipeline follows MLOps best practices. For improvements:
1. Maintain modular architecture
2. Add comprehensive tests
3. Follow configuration-driven approach
4. Ensure backward compatibility

---

**Ready to train your Whisper model?** 

```bash
chmod +x run_training.sh && ./run_training.sh
```

🎯 **Monitor your training at: http://localhost:5000**