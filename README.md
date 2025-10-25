# UIT Data Science Challenge 2025

## Track B – Evaluating LLM Robustness Against Noise & Hallucination in Vietnamese Fact-Checking

How can we build trustworthy Large Language Models (LLMs) that resist noise and minimize hallucination in real-world Vietnamese applications?

## Challenge Overview

In this track, teams must develop solutions to improve LLM robustness and reliability in Fact-Checking, specifically for Question Answering (QA) and Dialogue Tasks.

- Noise Robustness: The model should maintain consistent performance when inputs include:

    - Typos or grammatical errors

    - Ambiguous queries

    - Adversarial prompts designed to confuse the model

- Hallucination Reduction: The model must limit generating:

    - Unsupported or incorrect claims

    - Fabricated facts not grounded in the input

The evaluation metric is Macro-F1, calculated over three labels:

- no – no hallucination

- intrinsic – contradicts provided input

- extrinsic – introduces unsupported external information

This challenge aims to build practical Vietnamese AI systems that are robust, safe, and applicable in domains such as news verification, healthcare, finance, and enterprise AI.

## Idea

This project focuses on classifying whether a model response to a prompt (given some context) is:

(i) no — not hallucinated / consistent with context,

(ii) intrinsic — contradictory to the provided context,

(iii) extrinsic — introduces unsupported external facts.

The approach fuses three submodules:

The approach fuses three submodules:

1. **NLI-like head**: models alignment between context and response (context ↔ response).

2. **Coverage head**: computes forward/backward alignment scores as a regression-like signal.

3. **Prompt head**: models alignment between prompt and response (prompt ↔ response).

- Backbone: an autoregressive/encoder transformer loaded via Hugging Face AutoModel (default: uitnlp/CafeBERT). Tokenization keeps prompt, context, response in separate segments via token_type_ids.

## Requirements & Setup

Minimum environment (as implied by imports):

- Python 3.8+

- CUDA-compatible GPU recommended

- Libraries:

    - torch
    - transformers
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - tqdm
    - (optional) Google Colab / Drive if you use the BASE_DIR paths

Example pip install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```
The code uses AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True). Ensure network access to Hugging Face and have the model accessible.

## Key Features

- Custom `HallucinationDataset` handles tokenization and building `input_ids`, `token_type_ids`, and masks for three segments.

- **HallucinationModel**:
    - Pretrained transformer backbone

    - BiLSTM + TransformerEncoder for contextual interactions

    - NLI head (classification), Coverage regressor, Prompt head (classification)

    - Bilinear fusion of latent vectors and a final fusion classifier

- **Training**: 
    - Stratified K-Fold (default 5 folds)

    - Weighted cross-entropy for class imbalance + label smoothing

    - Composite loss: classifier + α * nli + β * coverage + γ * prompt

    - Mixed precision (torch.cuda.amp) and gradient accumulation

    - Early stopping per fold

- **Evaluation**: Macro-F1, classification report, confusion matrix visualizations

## Model Architecture

**Backbone**: AutoModel.from_pretrained(CFG.model_name) → last hidden states.

**Segment masks**: built using token_type_ids (0=context, 1=response, 2=prompt).

**Interaction**:
- Compute token-level alignment matrix e = H_seg1 @ H_seg2.T

- Compute attentive aligned token representations (tilde_H = softmax(e) @ H_other)

- Build four-way token features: [H, tilde_H, H - tilde_H, H * tilde_H]

- Pass through BiLSTM → Transformer encoder → pooled (avg + max) → MLP heads

**Heads**:
- NLI classifier (3 classes)

- Coverage regressor (scalar)

- Prompt classifier (3 classes)

**Fusion**:
- 3 latent vectors + 3 bilinear interactions → concatenated → final MLP → output logits (3 classes)

## Training Pipeline

- Create HallucinationDataset for train/val folds (tokenizes prompt/context/response preserving segment ids).

- Use StratifiedKFold (default n_splits=5) on combined train+val to perform cross-validation.

- For each fold:

    - Initialize model, optimizer (AdamW), learning rate scheduler (get_linear_schedule_with_warmup), GradScaler.

    - Compute class weights from training fold and use them in CrossEntropyLoss.

    - Train for up to num_train_epochs with:

        - Mixed precision

        - Gradient accumulation

        - Clip gradients

        - Evaluate on validation after each epoch

        - Early stopping based on Macro-F1

    - Save best model state (keeps the state of the overall best fold across folds)

- After K-Fold: reload best saved state, evaluate on the validation subset corresponding to that best state and print final metrics.  