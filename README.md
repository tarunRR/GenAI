# GenAI - Generative AI Projects & Learnings

A comprehensive repository exploring Generative AI concepts with practical implementation projects, including fine-tuning of large language models for domain-specific tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Projects](#projects)
- [Dependencies](#dependencies)
- [Usage](#usage)

## 🎯 Overview

This repository contains hands-on implementations and explorations of state-of-the-art generative AI techniques, including:

- **Fine-tuning Large Language Models**: Using parameter-efficient methods (LoRA) to adapt models for specific domains
- **Domain-specific Applications**: Insurance claims processing and medical terminology understanding
- **Model Optimization**: Techniques for reducing computational costs while maintaining performance

## 📁 Project Structure

```
GenAI/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── Notebooks/
    ├── test_notebook.ipynb   # Phi-2 fine-tuning for insurance claims
    └── phi2-insurance/       # Output directory for fine-tuned models
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda
- macOS, Linux, or Windows (with CUDA for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aditibawara/GenAI.git
cd GenAI
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## 📚 Projects

### Phi-2 Insurance Domain Fine-tuning

Fine-tune Microsoft's Phi-2 model on insurance claims data using LoRA (Low-Rank Adaptation) for efficient parameter updates.

**Notebook**: `Notebooks/test_notebook.ipynb`

**Key Features:**
- Uses the lightweight Phi-2 model (2.7B parameters)
- LoRA adaptation for parameter-efficient fine-tuning
- Dataset: Medical claims denial explanations and appeal responses
- Optimized for Mac (MPS), GPU (CUDA), and CPU execution
- Trained on insurance-specific instruction-response pairs

**Sample Use Cases:**
- Explaining insurance claim denials
- Drafting medical appeals
- Understanding authorization requirements

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.1 | Deep learning framework |
| `transformers` | 4.36.2 | Pre-trained models and tokenizers |
| `peft` | 0.3.0 | Parameter-efficient fine-tuning (LoRA) |
| `datasets` | 2.14.7 | Dataset management and processing |
| `accelerate` | 0.28.0 | Distributed training utilities |
| `bitsandbytes` | 0.39.0 | Optimized operations |
| `pandas` | 2.1.4 | Data manipulation |
| `scikit-learn` | 1.3.2 | ML utilities |
| `requests` | 2.31.0 | HTTP library |
| `python-dotenv` | 0.21.0 | Environment variable management |
| `playwright` | 1.41.2 | Web automation |

## 💻 Usage

### Running the Fine-tuning Notebook

1. Start Jupyter with the notebook:
```bash
jupyter notebook Notebooks/test_notebook.ipynb
```

2. Execute cells in order to:
   - Load and prepare the insurance claims dataset
   - Initialize the Phi-2 model and tokenizer
   - Apply LoRA configuration for efficient fine-tuning
   - Train the model on domain-specific data
   - Generate responses for new insurance queries

### Key Workflow Steps

1. **Data Preparation**: Format instruction-response pairs for causal language modeling
2. **Model Loading**: Load Phi-2 and apply LoRA adapters
3. **Training**: Fine-tune with optimized settings for consumer hardware
4. **Inference**: Generate domain-specific responses with the trained model

### Example: Generate Insurance Claim Response

```python
prompt = """### Instruction:
Explain denial for missing clinical documentation.

### Response:
"""

response = generate(prompt)
print(response)
```

## 🛠️ Hardware Considerations

- **GPU Support**: Automatically detects CUDA, MPS (Apple), or falls back to CPU
- **Memory Efficient**: LoRA reduces trainable parameters from millions to thousands
- **Batch Size**: Set to 1 for limited memory environments; adjust based on available VRAM

## 📖 Learning Resources

This repository demonstrates:
- Transformer model fine-tuning
- Parameter-efficient adaptation techniques
- Dataset creation and preparation
- Multi-device training compatibility

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements.

---

**Last Updated**: February 2026
