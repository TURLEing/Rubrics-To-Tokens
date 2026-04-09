from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import Dataset

from openrlhf.datasets.utils import exist_and_not_none
from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    criteria_key="criteria",
    response_key="response",
    token_labels_key="token_labels",
    char_labels_key="char_labels",
    apply_chat_template=None,
) -> tuple:
    """Preprocess data for token-level value model training.
    
    Args:
        data: Raw data dictionary
        input_template: Template for formatting prompt (not used in current version)
        prompt_key: Key for prompt in data dict (deprecated, kept for compatibility)
        criteria_key: Key for criteria in data dict
        response_key: Key for response in data dict
        token_labels_key: Key for token-level labels in data dict
        char_labels_key: Key for character-level labels in data dict
        apply_chat_template: Whether to apply chat template
        
    Returns:
        tuple: (prompt, response, token_labels)
    """
    # Get criteria (required) and response
    if criteria_key not in data or data[criteria_key] is None:
        raise ValueError(f"'{criteria_key}' must be provided in data")
    
    criteria = data[criteria_key]
    response = data[response_key]
    
    # Build prompt with enhanced format (includes task instruction for better model understanding)
    # Note: question is kept as optional field in data but not used in prompt construction
    prompt = f"Identify which tokens in the response are relevant to the criteria.\n\nCriteria: {criteria}\n\nResponse:\n\n"
    
    # Get labels - support both token-level and character-level labels
    token_labels = None
    if token_labels_key in data and data[token_labels_key] is not None:
        token_labels = data[token_labels_key]
    elif char_labels_key in data and data[char_labels_key] is not None:
        # Character-level labels will be converted to token-level in __getitem__
        token_labels = data[char_labels_key]  # Will be processed later
    else:
        raise ValueError(f"Either '{token_labels_key}' or '{char_labels_key}' must be provided in data")

    return prompt, response, token_labels


def char_labels_to_token_labels(
    response_text: str,
    char_labels: List[int],
    tokenizer: Callable,
) -> List[int]:
    """Convert character-level labels to token-level labels.
    
    Args:
        response_text: Response text (not including prompt)
        char_labels: Character-level labels (0 or 1) for response, length should match len(response_text)
        tokenizer: Tokenizer to use
        
    Returns:
        List[int]: Token-level labels for response tokens
    """
    # Validate input
    if len(char_labels) != len(response_text):
        raise ValueError(
            f"char_labels length ({len(char_labels)}) must match response_text length ({len(response_text)})"
        )
    
    # Tokenize the response part with offset mapping
    try:
        tokenized = tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        # Fallback if tokenizer doesn't support return_offsets_mapping
        tokenized = tokenizer(response_text, add_special_tokens=False)
        # Use a simple heuristic: assume each token maps to roughly equal character ranges
        token_labels = []
        tokens = tokenized["input_ids"]
        chars_per_token = len(response_text) / max(len(tokens), 1)
        for i, token_id in enumerate(tokens):
            char_start = int(i * chars_per_token)
            char_end = int((i + 1) * chars_per_token)
            char_end = min(char_end, len(char_labels))
            if char_start < len(char_labels):
                token_chars = char_labels[char_start:char_end]
                token_label = 1 if any(label == 1 for label in token_chars) else 0
            else:
                token_label = 0
            token_labels.append(token_label)
        return token_labels
    
    # Get character offsets for each token
    offsets = tokenized["offset_mapping"]
    token_labels = []
    
    for start_char, end_char in offsets:
        if start_char == end_char == 0:
            # Special token (e.g., padding) or empty token
            token_labels.append(0)
            continue
            
        # Offsets are relative to response_text
        # Get labels for characters covered by this token
        if start_char < len(char_labels) and end_char <= len(char_labels):
            token_chars = char_labels[start_char:end_char]
            # If any character in the token is labeled as 1, label the token as 1
            token_label = 1 if any(label == 1 for label in token_chars) else 0
        else:
            # Out of bounds, default to 0
            token_label = 0
        
        token_labels.append(token_label)
    
    return token_labels


class TokenLevelValueDataset(Dataset):
    """
    Dataset for token-level value model training.
    
    Each sample consists of:
    - prompt: User input/instruction
    - response: Model response
    - token_labels: Binary labels (0/1) for each token in response, indicating
      whether the token is relevant to the prompt requirement
    
    The dataset supports two label formats:
    1. Token-level labels: Direct labels for each token in response
    2. Character-level labels: Labels for each character in response (will be converted to token-level)
    
    Args:
        dataset: HuggingFace dataset or similar
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        strategy: Training strategy (for accessing args)
        input_template: Template for formatting prompt
        prompt_key: Key for prompt in data dict (default: "prompt")
        response_key: Key for response in data dict (default: "response")
        token_labels_key: Key for token-level labels (default: "token_labels")
        char_labels_key: Key for character-level labels (default: "char_labels")
        label_format: "token" or "char" (default: "token")
        num_processors: Number of processors for parallel processing
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        prompt_key="prompt",  # Deprecated, kept for compatibility
        criteria_key="criteria",
        response_key="response",
        token_labels_key="token_labels",
        char_labels_key="char_labels",
        label_format="token",  # "token" or "char"
        apply_chat_template=False,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.input_template = input_template
        self.prompt_key = prompt_key  # Deprecated
        self.criteria_key = criteria_key
        self.response_key = response_key
        self.token_labels_key = token_labels_key
        self.char_labels_key = char_labels_key
        self.label_format = label_format
        self.apply_chat_template = apply_chat_template

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Process dataset
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store processed data
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.labels = processed_dataset["labels"]  # Can be token-level or char-level
        self.label_formats = processed_dataset["label_format"]  # Track format per sample

    def process_data(self, data):
        """Process raw data into prompt, response, and labels."""
        prompt, response, labels = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.criteria_key,
            self.response_key,
            self.token_labels_key,
            self.char_labels_key,
            self.apply_chat_template if self.apply_chat_template else None,
        )

        # Check if prompt is too long
        if prompt:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # Filter samples where prompt is too long
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        # Determine label format
        label_format = self.label_format
        if isinstance(labels, list) and len(labels) > 0:
            # Try to infer format from label length vs response length
            if label_format == "auto":
                if len(labels) == len(response):
                    label_format = "char"
                else:
                    label_format = "token"

        return {
            "prompt": prompt,
            "response": response,
            "labels": labels,
            "label_format": label_format,
        }

    def __len__(self):
        return len(self.prompts)

    def _align_token_labels(
        self,
        prompt: str,
        response: str,
        labels: Union[List[int], None],
        label_format: str,
    ) -> tuple:
        """Align labels to tokenized sequence.
        
        Note: prompt format is "Criteria: {criteria}\n\nResponse: "
        
        Returns:
            tuple: (token_labels, response_start_idx, response_end_idx)
                - token_labels: Labels aligned to full sequence (prompt + response)
                - response_start_idx: Token index where response starts (after "Criteria: {criteria}\n\nResponse: ")
                - response_end_idx: Token index where response ends (exclusive)
        """
        # Tokenize prompt separately to get its length
        # prompt = "评估标准: {criteria}\n\n回答: "
        prompt_tokenized = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )
        prompt_len = len(prompt_tokenized["input_ids"])

        # Tokenize full sequence (prompt + response)
        full_text = prompt + response
        full_tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        full_input_ids = full_tokenized["input_ids"]
        offsets = full_tokenized["offset_mapping"]
        seq_len = len(full_input_ids)

        # Find response start position in tokenized sequence
        # response_start_idx is the token index where response content starts
        response_start_idx = prompt_len

        # Convert labels to token-level if needed
        if label_format == "char" and labels is not None:
            # Convert character-level labels to token-level
            # labels should correspond to response characters
            token_labels_response = char_labels_to_token_labels(
                response, labels, self.tokenizer
            )
        elif label_format == "token" and labels is not None:
            token_labels_response = labels
        else:
            token_labels_response = []

        # Create full sequence labels (initialize with -1 for prompt/padding)
        token_labels_full = [-1] * seq_len

        # Align response labels to full sequence
        response_token_count = len(token_labels_response)
        if response_start_idx + response_token_count <= seq_len:
            for i, label in enumerate(token_labels_response):
                if response_start_idx + i < seq_len:
                    token_labels_full[response_start_idx + i] = label
        else:
            # Truncation occurred, only use labels that fit
            available_len = seq_len - response_start_idx
            for i in range(min(available_len, len(token_labels_response))):
                token_labels_full[response_start_idx + i] = token_labels_response[i]

        response_end_idx = min(response_start_idx + len(token_labels_response), seq_len)

        return token_labels_full, response_start_idx, response_end_idx

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        labels = self.labels[idx]
        label_format = self.label_formats[idx]

        # Align labels to tokenized sequence
        token_labels, response_start_idx, response_end_idx = self._align_token_labels(
            prompt, response, labels, label_format
        )

        # Prepare full text: prompt already contains "Criteria: {criteria}\n\nResponse: "
        # So we just concatenate prompt + response
        full_text = prompt + response
        # Remove trailing newlines and ensure EOS token
        full_text = full_text.rstrip("\n")
        if not full_text.endswith(self.tokenizer.eos_token):
            full_text += " " + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Keep 2D shape (1, seq_len) to be consistent with other datasets
        # This ensures zero_pad_sequences works correctly with torch.cat
        input_ids = tokenized["input_ids"]  # Shape: (1, seq_len)
        attention_mask = tokenized["attention_mask"]  # Shape: (1, seq_len)
        seq_len = input_ids.shape[1]

        # Ensure EOS token is properly set
        if seq_len > 0:
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = 1

        # Adjust token_labels length if needed (due to truncation or EOS)
        if len(token_labels) > seq_len:
            token_labels = token_labels[:seq_len]
        elif len(token_labels) < seq_len:
            token_labels = token_labels + [-1] * (seq_len - len(token_labels))

        # Create response mask (1 for response tokens, 0 for prompt/padding)
        # Keep 2D shape (1, seq_len) to match input_ids
        response_mask = torch.zeros(1, seq_len, dtype=torch.float32)
        if response_start_idx < seq_len:
            response_mask[0, response_start_idx:response_end_idx] = 1.0

        # Convert token_labels to tensor with 2D shape (1, seq_len)
        token_labels_tensor = torch.tensor(token_labels, dtype=torch.long).unsqueeze(0)

        return (
            input_ids,
            attention_mask,
            token_labels_tensor,
            response_mask,
            response_start_idx,
        )

    def collate_fn(self, item_list):
        """Collate function for batching."""
        input_ids_list = []
        attention_masks_list = []
        token_labels_list = []
        response_masks_list = []
        response_start_indices = []

        for input_ids, attention_mask, token_labels, response_mask, response_start_idx in item_list:
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            token_labels_list.append(token_labels)
            response_masks_list.append(response_mask)
            response_start_indices.append(response_start_idx)

        # Left padding (consistent with OpenRLHF style)
        padding_side = "left"
        pad_value = self.tokenizer.pad_token_id

        input_ids = zero_pad_sequences(input_ids_list, side=padding_side, value=pad_value)
        attention_masks = zero_pad_sequences(attention_masks_list, side=padding_side, value=0)
        token_labels = zero_pad_sequences(token_labels_list, side=padding_side, value=-1)
        response_masks = zero_pad_sequences(response_masks_list, side=padding_side, value=0)

        # With 2D input tensors from __getitem__, zero_pad_sequences should return 2D tensors (batch_size, seq_len)
        # This is consistent with other datasets (sft_dataset, reward_dataset)

        return (
            input_ids,
            attention_masks,
            token_labels,
            response_masks,
            response_start_indices,
        )

