# Paraphrase Generation with Text-to-Text Transfer Transformer

## 📌 Overview

This repository hosts the quantized version of the T5 model fine-tuned for Paraphrase Generation. The model has been trained on the chatgpt-paraphrases dataset from Hugging Face to enhance grammatical accuracy in given text inputs. The model is quantized to Float16 (FP16) to optimize inference speed and efficiency while maintaining high performance.

## 🏗 Model Details

- **Model Architecture:** t5-small
- **Task:** Paraphrase Generation
- **Dataset:** Hugging Face's `chatgpt-paraphrases`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/t5-paraphrase-generation"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)
```

### Grammar Correction Inference

```python
paraphrase_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
test_text = "The quick brown fox jumps over the lazy dog"

# Generate paraphrases
results = paraphrase_pipeline(
    test_text,
    max_length=256,
    truncation=True,
    num_return_sequences=5,
    do_sample=True,
    top_k=50,
    temperature=0.7
)

print("Original Text:", test_text)
print("\nParaphrased Outputs:")

for i, output in enumerate(results):
    generated_text = output["generated_text"] if isinstance(output, dict) else str(output)
    print(f"{i+1}. {generated_text.strip()}")
```

# 📊 ROUGE Evaluation Results
 
After fine-tuning the **T5-Small** model for paraphrase generation, we obtained the following **ROUGE** scores:

| **Metric**  | **Score**  | **Meaning** |
|-------------|-----------|-------------|
| **ROUGE-1** | **0.7777** (~78%) | Measures overlap of **unigrams (single words)** between the reference and generated summary. |
| **ROUGE-2** | **0.5** (~50%) | Measures overlap of **bigrams (two-word phrases)**, indicating coherence and fluency. |
| **ROUGE-L** | **0.7777** (~78%) | Measures **longest matching word sequences**, testing sentence structure preservation. |
| **ROUGE-Lsum** | **0.7777** (~78%) | Similar to ROUGE-L but optimized for summarization tasks. |


## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle with highly ambiguous sentences.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different writing styles and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
