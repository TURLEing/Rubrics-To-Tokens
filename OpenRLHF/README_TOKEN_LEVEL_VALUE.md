# Token-Level Value Model Training

This guide explains how to train a **Token-Level Value Model** using the OpenRLHF framework. The model outputs a 0/1 label for each token in the response portion of a concatenated ''criteria + response'' sequence, indicating whether the token is relevant to the criteria.


## Overview

The Token-Level Value Model is designed for fine-grained evaluation:

- **Input**: A concatenated sequence of criteria (evaluation standard) + response (model output)
  - Prompt format: `"Criteria: {criteria}\n\nResponse: "`
- **Output**: A binary classification label (0 or 1) for each token in the response portion
  - `1` indicates the token is relevant to the criteria
  - `0` indicates the token is irrelevant to the criteria

This fine-grained evaluation can be used for:
- Identifying which parts of a response satisfy specific evaluation criteria
- Detecting irrelevant or redundant content
- Providing more precise feedback signals for subsequent model optimization

**Note**: The data may include a `question` field, but the current version does not use it. The model judges response token relevance based solely on the criteria.

## Quick Start

### 1. Prepare Data

First, prepare your training data in the following format:

```json
{
    "question": "Please explain what machine learning is.",
    "criteria": "Does the answer include specific examples?",
    "response": "Machine learning is a branch of artificial intelligence that enables computers to learn from data. For example, recommendation systems use machine learning to predict which movies a user might like.",
    "token_labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

**Field descriptions**:
- `question`: (Optional) The original question. Not used in the current version, but can be kept in the data.
- `criteria`: (Required) The evaluation standard used to determine which response tokens are relevant.
- `response`: (Required) The model's response.
- `token_labels`: (Required) Labels for each response token. The length must equal the number of tokens after tokenizing the response.

**Important**: The length of `token_labels` must equal the number of tokens after tokenizing the response.

### 2. Run Training

```bash
deepspeed --num_gpus=8 openrlhf/cli/train_token_level_value.py \
    --pretrain /path/to/your/pretrained/model \
    --dataset /path/to/your/data_with_token_label.jsonl \
    --save_path ./ckpt \
    --max_len 4096 \
    --micro_train_batch_size 4 \
    --train_batch_size 256 \
    --learning_rate 9e-6 \
    --max_epochs 2 \
    --bf16 \
    --zero_stage 2 \
    --use_tensorboard ./log
```

## Data Format

### Format 1: Token-Level Labels (Recommended)

```json
{
    "question": "Please explain what machine learning is.",
    "criteria": "Does the answer include specific examples?",
    "response": "Machine learning is a branch of artificial intelligence...",
    "token_labels": [1, 1, 0, 1, ...]  // Label for each response token
}
```

**Field descriptions**:
- `question`: (Optional) The original question. Not used in the current version, but can be kept for analysis.
- `criteria`: (Required) The evaluation standard used to determine which response tokens are relevant.
- `response`: (Required) The model's response.
- `token_labels`: (Required) A list of integers, each being 0 or 1. The length must equal the number of tokens after tokenizing the response.
  - `1` indicates the token is relevant to the criteria
  - `0` indicates the token is irrelevant to the criteria

**Prompt construction**:
Internally, the model constructs the data in the following format:
```
Criteria: {criteria}

Response: {response}
```

### Format 2: Character-Level Labels

If you have character-level annotations, you can also use them:

```json
{
    "question": "Please explain what machine learning is.",
    "criteria": "Does the answer include specific examples?",
    "response": "Machine learning is a branch of artificial intelligence...",
    "char_labels": [1, 1, 1, 0, ...]  // Label for each response character
}
```

When using character-level labels, set `--label_format char`. The dataset class will automatically convert character-level labels to token-level labels.

### Verifying Label Alignment

Before training, it is recommended to verify that the label count is correct:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model-path")
response = "Your response text"
tokenized = tokenizer(response, add_special_tokens=False)
num_tokens = len(tokenized["input_ids"])

# Ensure the length of token_labels equals num_tokens
assert len(token_labels) == num_tokens, f"Token count mismatch: {len(token_labels)} vs {num_tokens}"
```

## Model Architecture

The Token-Level Value Model is built on a pretrained language model with an added value head:

```
[Base LLM] → [Hidden States] → [Value Head] → [Token Logits] → [Sigmoid] → [0/1 Labels]
```

- **Base LLM**: A pretrained language model (e.g., LLaMA, ChatGLM, etc.)
- **Value Head**: A linear layer that maps hidden states to scalar logits
- **Output**: Logits at each token position, converted to 0-1 probabilities via sigmoid

## Training Pipeline

### 1. Data Loading

Use `TokenLevelValueDataset` to load data:

```python
from openrlhf.datasets import TokenLevelValueDataset

dataset = TokenLevelValueDataset(
    dataset=your_dataset,
    tokenizer=tokenizer,
    max_length=512,
    strategy=strategy,
    criteria_key="criteria",
    response_key="response",
    token_labels_key="token_labels",
    label_format="token",  # or "char"
)
```

### 2. Model Initialization

Use `get_llm_for_token_level_value` to create the model:

```python
from openrlhf.models import get_llm_for_token_level_value

model = get_llm_for_token_level_value(
    model_name_or_path="your-model-path",
    bf16=True,
    init_value_head=True,
    value_head_prefix="score",
)
```

### 3. Training

Use `TokenLevelValueTrainer` for training:

```python
from openrlhf.trainer import TokenLevelValueTrainer

trainer = TokenLevelValueTrainer(
    model=model,
    strategy=strategy,
    optim=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    scheduler=scheduler,
    tokenizer=tokenizer,
    max_epochs=1,
)

trainer.fit(args, consumed_samples=0, num_update_steps_per_epoch=num_steps)
```

### 4. Loss Computation

Training uses Binary Cross-Entropy Loss, computed only on response tokens:

```python
# Compute loss only where response_mask > 0 and token_labels >= 0
valid_mask = (token_labels >= 0) & (response_mask > 0)
loss = F.binary_cross_entropy_with_logits(
    logits[valid_mask],
    token_labels[valid_mask].float()
)
```

## Usage Examples

### Full Training Example

```bash
# Single GPU training
python -m openrlhf.cli.train_token_level_value \
    --pretrain meta-llama/Llama-2-7b-hf \
    --dataset ./data/train.json \
    --save_path ./ckpt/token_level_value \
    --max_len 512 \
    --micro_train_batch_size 2 \
    --train_batch_size 64 \
    --learning_rate 9e-6 \
    --max_epochs 1 \
    --bf16 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --criteria_key criteria \
    --response_key response \
    --token_labels_key token_labels \
    --label_format token

# Multi-GPU training (with DeepSpeed)
deepspeed --num_gpus=4 openrlhf/cli/train_token_level_value.py \
    --pretrain meta-llama/Llama-2-7b-hf \
    --dataset ./data/train.json \
    --save_path ./ckpt/token_level_value \
    --max_len 512 \
    --micro_train_batch_size 1 \
    --train_batch_size 128 \
    --learning_rate 9e-6 \
    --max_epochs 1 \
    --bf16 \
    --zero_stage 2 \
    --gradient_checkpointing
```

### Fine-tuning with LoRA

```bash
python -m openrlhf.cli.train_token_level_value \
    --pretrain meta-llama/Llama-2-7b-hf \
    --dataset ./data/train.json \
    --save_path ./ckpt/token_level_value_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules q_proj v_proj k_proj o_proj \
    --max_len 512 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --learning_rate 1e-4 \
    --max_epochs 3 \
    --bf16 \
    --zero_stage 2
```

### Using Character-Level Labels

```bash
python -m openrlhf.cli.train_token_level_value \
    --pretrain meta-llama/Llama-2-7b-hf \
    --dataset ./data/train.json \
    --save_path ./ckpt/token_level_value \
    --char_labels_key char_labels \
    --label_format char \
    --max_len 512 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --learning_rate 9e-6 \
    --max_epochs 1 \
    --bf16 \
    --zero_stage 2
```

## Key Parameters

### Model Parameters

- `--pretrain`: Pretrained model path (required)
- `--value_head_prefix`: Name prefix for the value head (default: "score")
- `--lora_rank`: LoRA rank (0 means LoRA is not used)
- `--lora_alpha`: LoRA alpha parameter
- `--target_modules`: Target modules for LoRA

### Data Parameters

- `--dataset`: Training dataset path
- `--eval_dataset`: Evaluation dataset path (optional)
- `--criteria_key`: Key name for the criteria field in the dataset (default: "criteria")
- `--response_key`: Key name for the response field in the dataset (default: "response")
- `--token_labels_key`: Key name for the token_labels field in the dataset (default: "token_labels")
- `--char_labels_key`: Key name for the char_labels field in the dataset (default: "char_labels")
- `--label_format`: Label format, "token" or "char" (default: "token")
- `--max_len`: Maximum sequence length (default: 512)

**Note**: The data may include a `question` field, but the current version does not use it. The model judges response token relevance based solely on `criteria`.

### Training Parameters

- `--learning_rate`: Learning rate (default: 9e-6)
- `--max_epochs`: Maximum number of training epochs (default: 1)
- `--micro_train_batch_size`: Micro batch size (default: 1)
- `--train_batch_size`: Global batch size (default: 128)
- `--lr_warmup_ratio`: Learning rate warmup ratio (default: 0.03)
- `--lr_scheduler`: Learning rate scheduler (default: "cosine_with_min_lr")

### DeepSpeed Parameters

- `--zero_stage`: ZeRO stage (0, 1, 2, 3)
- `--bf16`: Use bfloat16 precision
- `--gradient_checkpointing`: Enable gradient checkpointing

## Inference

After training, you can use the model for inference:

```python
from openrlhf.models import get_llm_for_token_level_value
from transformers import AutoTokenizer
import torch

# Load model
model = get_llm_for_token_level_value(
    model_name_or_path="./ckpt/token_level_value",
    bf16=True,
)
tokenizer = AutoTokenizer.from_pretrained("./ckpt/token_level_value")

# Prepare input
criteria = "Does the answer include specific examples?"
response = "Machine learning is a branch of artificial intelligence. For example, recommendation systems use machine learning to predict which movies a user might like."

# Build prompt (consistent with training format)
prompt = f"Criteria: {criteria}\n\nResponse: "
full_text = prompt + response

# Tokenize
inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Compute response start position
prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
response_start_idx = len(prompt_inputs["input_ids"][0])

# Create response mask
response_mask = torch.zeros(inputs["input_ids"].shape[1])
response_mask[response_start_idx:] = 1.0

# Inference
model.eval()
with torch.no_grad():
    logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        response_mask=response_mask.unsqueeze(0),
    )

    # Convert to probabilities
    probs = torch.sigmoid(logits)

    # Get predictions for the response portion
    response_probs = probs[0, response_start_idx:]
    predictions = (response_probs > 0.5).long().tolist()

    print(f"Criteria: {criteria}")
    print(f"Response: {response}")
    print(f"Response tokens: {len(predictions)}")
    print(f"Predictions: {predictions}")
```

## FAQ

### Q1: How do I determine the length of token_labels?

A: Tokenize the response first to confirm the token count:

```python
tokenizer = AutoTokenizer.from_pretrained("your-model-path")
tokenized = tokenizer(response, add_special_tokens=False)
num_tokens = len(tokenized["input_ids"])
# The length of token_labels should equal num_tokens
```

### Q1.5: Why use criteria instead of question?

A: Based on the task requirements, the model needs to determine which tokens in the response are relevant to a specific evaluation standard (criteria), rather than the entire question.

Criteria are evaluation standards extracted from the question. Using criteria allows the model to focus on specific evaluation dimensions. The `question` field can be kept in the data for analysis, but it is not used during training.

### Q2: How to handle alignment issues caused by tokenization?

A: The dataset class handles alignment automatically. If using character-level labels, they are automatically converted to token-level labels. Recommendations:
- Use token-level labels (more precise)
- Consider tokenization results during annotation
- Use `return_offsets_mapping=True` for precise alignment

### Q3: What if the loss doesn't decrease during training?

A: Check the following:
- Whether labels are correctly aligned
- Whether the learning rate is appropriate (try 1e-5 to 1e-6)
- Whether data quality is sufficient
- Whether loss is computed only on the response portion (check response_mask)

### Q4: How to use a custom dataset format?

A: You can specify field names via parameters:

```bash
--criteria_key your_criteria_key \
--response_key your_response_key \
--token_labels_key your_labels_key
```

**Note**: The prompt format is fixed as `"Criteria: {criteria}\n\nResponse: "`. To customize the format, you need to modify the dataset class code.

### Q5: Which models are supported?

A: All HuggingFace Transformers-compatible models are supported, including:
- LLaMA / LLaMA-2
- ChatGLM
- Qwen
- Baichuan
- And more

## References

- [OpenRLHF Main Documentation](../README.md)
- [Dataset Usage Guide](./examples/token_level_value_dataset_example.md)
- [Model Implementation](../openrlhf/models/model.py)
- [Trainer Implementation](../openrlhf/trainer/token_level_value_trainer.py)

## License

Consistent with the OpenRLHF project.
