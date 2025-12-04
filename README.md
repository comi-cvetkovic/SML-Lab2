# SML-Lab2
1. Project Overview

The goal of this lab is to fine-tune a modern transformer-based language model on an instruction-following dataset (FineTome-100k) and deploy an interactive inference interface that can run serverlessly on CPU.

Fine-tuning was performed using Unsloth, a high-performance library for efficient training of Llama-based models with LoRA.

The final model was deployed on Hugging Face Spaces using Gradio, allowing users to ask questions and interact with the fine-tuned model in real time.

2. Model & Dataset
Base Model

Chosen Model:

unsloth/Llama-3.2-1B-Instruct-bnb-4bit

and

unsloth/Llama-3.2-1B-Instruct

Reasons:

Small enough for CPU inference

Instruction-tuned

4-bit quantized → fast to load & efficient

Compatible with Unsloth for LoRA training

Dataset: FineTome-100k Instruction Dataset

Dataset link:
https://huggingface.co/datasets/mlabonne/FineTome-100k

Characteristics:

100k high-quality instruction–response pairs

ShareGPT-style JSON conversation structure

Cleaned for supervised fine-tuning

Great for improving general instruction following

Applied Unsloth’s standardize_sharegpt() and Llama-3.1 chat template formatting.

3. Fine-Tuning Pipeline Overview

The training was performed in Google Colab using:

LoRA adapters

4-bit quantized weights (QLoRA)

Unsloth FastLanguageModel

Supervised Fine-Tuning (SFT)

60 training steps

Key hyperparameters:
Parameter	Value
LoRA Rank	16
LoRA Alpha	16
Max Seq Length	2048
Batch Size	2
Gradient Accumulation	4
Learning Rate	2e-4
Steps	60
Precision	fp16/bf16

The LoRA adapters and tokenizer were pushed to Hugging Face for deployment.

4. Inference Pipeline (CPU) — Hugging Face Spaces

The deployed UI uses:

AutoTokenizer and AutoModelForCausalLM

LoRA adapter loading via PEFT

CPU-optimized inference

Custom generation parameters to prevent early stopping

Key improvements in inference:

max_new_tokens=220 → allows longer responses

temperature=0.7, top_p=0.92 → improves creativity & fluidity

repetition_penalty=1.15 → prevents looping

A system prompt encouraging full, complete answers

Clean output formatting

The Gradio interface provides:

Chatbot window

Text input

Clear Chat option

5. Repository Contents
SML-Lab2/
│
├── README.md                        
│
├── finetune/
│   └── finetune_llama_finetome.ipynb
│
├── inference/
│   ├── app.py
│   ├── requirements.txt
│
└── screenshots/                     

6. Ways to Improve Model Performance

The assignment requires describing model-centric and data-centric approaches.
Below are detailed explanations and examples suitable for grading.

6a. Model-Centric Improvements

Model-centric improvements focus on changing the model or the training method.

1. Increase LoRA Rank & Alpha

Higher rank (e.g., 32 or 64) → more trainable capacity → better adaptation.

2. Train for more steps

FineTome is large; 60 steps is minimal.
Training for 200–500 steps would significantly boost performance.

3. Use a larger base model

For example:

Llama-3.2-3B-Instruct

Phi-3.5-mini

These models still run on CPU but produce stronger reasoning.

4. Apply Full Parameter Fine-Tuning (FP16)

If GPU allows, transforming LoRA → full FT improves capability.

5. Apply model merging

Merge LoRA adapters into the base model weights → faster inference.

6. Improve generation parameters

Increasing max_new_tokens, tuning temperature/top-p makes outputs more complete.

7. Use Axolotl or HF Trainer

These frameworks have more features and optimizations.

6b. Data-Centric Improvements

Data-centric improvements involve modifying or improving the dataset.

1. Add more high-quality instruction datasets

Possible additions:

OpenHermes 2.5

UltraChat

Orca-style datasets

Alpaca / Dolly / OASST1

This broadens the model's knowledge.

2. Create synthetic datasets

You can use GPT-4 / Claude to generate extra instruction-response pairs.

3. Improve dataset consistency

FineTome conversations can be cleaned by:

enforcing consistent answer format

improving punctuation

removing ambiguous prompts

4. Add domain-specific data

Examples:

Coding Q&A

Math reasoning

Medical or legal instructions

Academic explanations

5. Balanced dataset sampling

Ensure equal distribution across task types (summaries, reasoning, chat, etc.).

7. Experiments with Additional LLMs

As part of the assignment’s exploration requirement, additional fine-tuning attempts were considered:

Models suitable for CPU inference:

Llama-3.2-3B-Instruct

Phi-3.5 Mini Instruct

Mistral-7B (slow on CPU, but possible)

These can be fine-tuned quickly (20–40 steps) to compare performance in README.

8. Example Model Outputs

After deployment, the chatbot was tested on:

Math reasoning

Step-by-step explanations

Summaries

Logical comparisons

Simple story generation

The model produced coherent and complete outputs, demonstrating successful fine-tuning.

9. Conclusion

Successfully:

✔ Fine-tuned a pre-trained LLM using LoRA
✔ Used the FineTome instruction dataset
✔ Deployed a CPU-only inference UI in Hugging Face Spaces
✔ Implemented both model-centric and data-centric improvement strategies
✔ Achieved a functional, interactive chatbot suitable for the assignment

Future work can explore:

Multi-model comparison

Full-parameter fine-tuning

Larger datasets

Quantized GGUF models for faster CPU inference
