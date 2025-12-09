import torch
from torch import nn
from typing import Optional, Union
import re
from dataclasses import dataclass, field


@dataclass
class SequenceSegment:
    """Represents a segment in the sequence - either tokens or embeddings."""
    is_embedding: bool
    tokens: Optional[list[int]] = None
    placeholder_key: Optional[str] = None
    is_label: bool = False  # True if this segment should be included in loss


class MultiModalBatchProcessor:
    """
    Processor for multi-modal LLM inputs that handles:
    - Text tokenization with chat templates
    - Embedding placeholder replacement  
    - Proper concatenation of text tokens and external embeddings
    - Label masking for training (only compute loss on assistant response)
    
    Placeholders are strings matching pattern <...> (e.g., <emb_1>, <transaction>)
    """
    
    PLACEHOLDER_PATTERN = re.compile(r'^<[^<>]+>$')
    
    def __init__(
        self,
        tokenizer,
        system_prompt: str = "You are a helpful assistant.",
        max_length: Optional[int] = 2048,
        padding_side: str = "right",
        ignore_index: int = -100,
    ):
        """
        Initialize the processor.
        
        Args:
            tokenizer: HuggingFace tokenizer with chat template support
            system_prompt: System prompt to use for all samples
            max_length: Maximum sequence length (None for no limit)
            padding_side: "left" or "right" padding
            ignore_index: Label index to ignore in loss computation (default: -100)
        """
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.padding_side = padding_side
        self.ignore_index = ignore_index
        
        # Ensure tokenizer has required tokens
        self.pad_token_id = self._get_pad_token_id()
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        
        # Storage for batch data
        self._inputs: Optional[list[list[str]]] = None
        self._labels: Optional[list[str]] = None
        self._transactions: Optional[dict[str, torch.Tensor]] = None
        self._batch_size: int = 0
        
    def _get_pad_token_id(self) -> int:
        """Get pad token id, falling back to eos if not set."""
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        raise ValueError("Tokenizer must have pad_token_id or eos_token_id")
    
    def is_placeholder(self, text: str) -> bool:
        """Check if a string is an embedding placeholder like <emb_1>."""
        return bool(self.PLACEHOLDER_PATTERN.match(text.strip()))
    
    def process_inputs(self, inputs: list[list[str]], labels: list[str]) -> "MultiModalBatchProcessor":
        """
        Process and store input sequences and labels.
        
        Args:
            inputs: List of input sequences (batch_size,). Each sequence is a 
                   list of strings that can be regular text or placeholders like <emb_1>.
                   Elements are concatenated in order to form the user message.
            labels: List of target responses (batch_size,) - assistant outputs
            
        Returns:
            self for method chaining
            
        Example:
            >>> processor.process_inputs(
            ...     inputs=[["Analyze this:", "<trans_emb>", "and respond"]],
            ...     labels=["The transaction shows..."]
            ... )
        """
        if len(inputs) != len(labels):
            raise ValueError(
                f"inputs and labels must have same batch size, "
                f"got {len(inputs)} and {len(labels)}"
            )
        
        self._inputs = inputs
        self._labels = labels
        self._batch_size = len(inputs)
        
        return self
    
    def add_transactions(self, transactions: dict[str, torch.Tensor]) -> "MultiModalBatchProcessor":
        """
        Add embedding transactions for placeholders.
        
        Args:
            transactions: Dict mapping placeholder names (e.g., "<emb_1>") to embeddings.
                         Each value has shape (batch_size, emb_dim).
                         
        Returns:
            self for method chaining
            
        Example:
            >>> processor.add_transactions({
            ...     "<trans_emb>": torch.randn(batch_size, 768),
            ...     "<user_emb>": torch.randn(batch_size, 768),
            ... })
        """
        if self._batch_size > 0:
            for key, tensor in transactions.items():
                if tensor.shape[0] != self._batch_size:
                    raise ValueError(
                        f"Transaction '{key}' has batch size {tensor.shape[0]}, "
                        f"expected {self._batch_size}"
                    )
        
        self._transactions = transactions
        return self
    
    def reset(self) -> None:
        """Clear all stored data for next batch."""
        self._inputs = None
        self._labels = None
        self._transactions = None
        self._batch_size = 0
    
    def _get_chat_template_parts(self) -> dict[str, list[int]]:
        """
        Extract chat template token sequences for different parts.
        
        Returns:
            Dict with tokenized template parts:
            - system_prefix: tokens before system content
            - system_suffix: tokens after system content
            - user_prefix: tokens before user content  
            - user_suffix: tokens after user content
            - assistant_prefix: tokens before assistant content
            - assistant_suffix: tokens after assistant content (e.g., </s>)
        """
        # Use dummy content to extract template structure
        dummy_messages = [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
            {"role": "assistant", "content": "ASSISTANT_MARKER"},
        ]
        
        # Get full template as text
        full_text = self.tokenizer.apply_chat_template(
            dummy_messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Get template without assistant content for comparison
        messages_no_assistant = [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            messages_no_assistant,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return {
            "full_template": full_text,
            "prefix_template": prefix_text,
        }
    
    def _tokenize_text(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize text without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def _build_sequence_segments(
        self,
        user_parts: list[str],
        assistant_response: str,
    ) -> list[SequenceSegment]:
        """
        Build sequence as list of segments (tokens or embedding placeholders).
        
        Args:
            user_parts: List of user input parts (text or placeholders)
            assistant_response: The target assistant response
            
        Returns:
            List of SequenceSegment objects representing the full sequence
        """
        segments = []
        
        # Build messages for tokenization
        # First, get the system + user prefix tokens
        system_user_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ""},  # Empty placeholder
        ]
        
        # Get template with empty user content to find where user content goes
        template_with_empty_user = self.tokenizer.apply_chat_template(
            system_user_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Find position of empty user content by using markers
        system_only = [{"role": "system", "content": self.system_prompt}]
        
        # Tokenize system prompt with template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Get tokens up to user content
            system_with_user_prefix = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ""}
                ],
                tokenize=True,
                add_generation_prompt=False
            )
            
            # These are the tokens before user content begins
            # We need to figure out where user content would be inserted
            # Use a marker approach
            marker = "<<<CONTENT_MARKER>>>"
            with_marker = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": marker}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Split around marker
            parts = with_marker.split(marker)
            user_prefix_text = parts[0]  # Text before user content
            user_suffix_text = parts[1] if len(parts) > 1 else ""  # Text after user content
            
            # Tokenize prefix (system + user turn start)
            user_prefix_tokens = self._tokenize_text(user_prefix_text)
            segments.append(SequenceSegment(
                is_embedding=False,
                tokens=user_prefix_tokens,
                is_label=False
            ))
            
            # Process user content parts
            for part in user_parts:
                part = part.strip()
                if not part:
                    continue
                    
                if self.is_placeholder(part):
                    # This is an embedding placeholder
                    segments.append(SequenceSegment(
                        is_embedding=True,
                        placeholder_key=part,
                        is_label=False
                    ))
                else:
                    # Regular text - tokenize it
                    part_tokens = self._tokenize_text(part)
                    if part_tokens:
                        segments.append(SequenceSegment(
                            is_embedding=False,
                            tokens=part_tokens,
                            is_label=False
                        ))
            
            # Add user suffix and assistant prefix
            # Get the assistant prefix using marker
            with_assistant_marker = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": marker}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Find text between user content end and assistant content start
            user_x_template = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "X"},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # The generation prompt gives us the assistant prefix
            assistant_prefix_text = user_x_template[len(user_prefix_text) + 1:]  # +1 for 'X'
            
            # Add tokens between user content and assistant content (if any)
            transition_tokens = self._tokenize_text(user_suffix_text + assistant_prefix_text)
            if transition_tokens:
                segments.append(SequenceSegment(
                    is_embedding=False,
                    tokens=transition_tokens,
                    is_label=False
                ))
            
            # Add assistant response tokens - THESE ARE LABELS
            response_tokens = self._tokenize_text(assistant_response)
            if response_tokens:
                segments.append(SequenceSegment(
                    is_embedding=False,
                    tokens=response_tokens,
                    is_label=True  # This is the target for training
                ))
            
            # Add end tokens (EOS) if the template adds them
            full_template = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": "Y"}
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Check if there's a suffix after assistant content
            assistant_end_idx = full_template.rfind("Y") + 1
            assistant_suffix = full_template[assistant_end_idx:]
            if assistant_suffix.strip():
                suffix_tokens = self._tokenize_text(assistant_suffix)
                if suffix_tokens:
                    segments.append(SequenceSegment(
                        is_embedding=False,
                        tokens=suffix_tokens,
                        is_label=True  # Include EOS in loss
                    ))
        
        return segments
    
    def _process_single_item(
        self,
        batch_idx: int,
        embed_layer: nn.Embedding,
        device: torch.device,
        emb_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single batch item into embeddings, labels, and attention mask.
        
        Returns:
            Tuple of (embeddings, labels, attention_mask) - each shape (seq_len,) or (seq_len, emb_dim)
        """
        user_parts = self._inputs[batch_idx]
        assistant_response = self._labels[batch_idx]
        
        # Build sequence segments
        segments = self._build_sequence_segments(user_parts, assistant_response)
        
        # Collect embeddings and labels
        all_embeddings = []
        all_labels = []
        
        for segment in segments:
            if segment.is_embedding:
                # Get embedding from transactions
                key = segment.placeholder_key
                if self._transactions is None or key not in self._transactions:
                    raise ValueError(
                        f"Placeholder '{key}' not found in transactions. "
                        f"Available: {list(self._transactions.keys()) if self._transactions else []}"
                    )
                
                emb = self._transactions[key][batch_idx]  # (emb_dim,)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)  # (1, emb_dim)
                
                all_embeddings.append(emb.to(device))
                # Embedding positions are masked in labels
                all_labels.extend([self.ignore_index] * emb.shape[0])
                
            else:
                # Token segment - convert to embeddings
                tokens = segment.tokens
                if not tokens:
                    continue
                    
                token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                token_embeds = embed_layer(token_tensor)  # (num_tokens, emb_dim)
                
                all_embeddings.append(token_embeds)
                
                if segment.is_label:
                    # These tokens should contribute to loss
                    all_labels.extend(tokens)
                else:
                    # Mask these tokens
                    all_labels.extend([self.ignore_index] * len(tokens))
        
        # Concatenate all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)  # (seq_len, emb_dim)
        else:
            embeddings = torch.zeros(0, emb_dim, device=device)
        
        # Convert labels to tensor
        labels = torch.tensor(all_labels, dtype=torch.long, device=device)
        
        # Attention mask is all 1s (we'll add padding later)
        attention_mask = torch.ones(len(all_labels), dtype=torch.long, device=device)
        
        return embeddings, labels, attention_mask
    
    def resolve_batch(
        self,
        embed_layer: nn.Embedding,
        device: Optional[Union[str, torch.device]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Resolve all inputs into final embeddings, attention masks, and labels.
        
        This method aggregates all processed inputs and transactions into
        the final format required for model training.
        
        Args:
            embed_layer: The embedding layer from the model (nn.Embedding)
                        Used to convert token IDs to embeddings
            device: Device to place output tensors on (default: same as embed_layer)
            
        Returns:
            dict with:
            - `inputs_embeds`: (batch_size, seq_len, emb_dim) - Input embeddings including
              chat template tokens, user input, embedded transactions, and assistant response.
              All placed in the correct order as specified in input lists.
            - `attention_mask`: (batch_size, seq_len) - Binary mask where 1 indicates
              real tokens and 0 indicates padding tokens.
            - `labels`: (batch_size, seq_len) - Target token IDs for training.
              All positions before assistant response are masked with -100.
              Only assistant response tokens (and EOS) contribute to loss.
              
        Raises:
            ValueError: If process_inputs hasn't been called or transactions are missing
            
        Example:
            >>> result = processor.resolve_batch(model.get_input_embeddings())
            >>> outputs = model(
            ...     inputs_embeds=result['inputs_embeds'],
            ...     attention_mask=result['attention_mask'],
            ...     labels=result['labels']
            ... )
        """
        # Validation
        if self._inputs is None or self._labels is None:
            raise ValueError(
                "Must call process_inputs() before resolve_batch()"
            )
        
        # Determine device
        if device is None:
            device = next(embed_layer.parameters()).device
        else:
            device = torch.device(device)
        
        # Get embedding dimension
        emb_dim = embed_layer.embedding_dim
        batch_size = self._batch_size
        
        # Process each item in the batch
        batch_embeddings = []
        batch_labels = []
        batch_attention_masks = []
        
        for batch_idx in range(batch_size):
            embeddings, labels, attention_mask = self._process_single_item(
                batch_idx=batch_idx,
                embed_layer=embed_layer,
                device=device,
                emb_dim=emb_dim,
            )
            batch_embeddings.append(embeddings)
            batch_labels.append(labels)
            batch_attention_masks.append(attention_mask)
        
        # Find max sequence length
        seq_lengths = [e.shape[0] for e in batch_embeddings]
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        
        # Apply max_length truncation if specified
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)
        
        # Initialize padded tensors
        padded_embeddings = torch.zeros(
            batch_size, max_seq_len, emb_dim, 
            dtype=batch_embeddings[0].dtype if batch_embeddings else torch.float32,
            device=device
        )
        padded_labels = torch.full(
            (batch_size, max_seq_len), 
            self.ignore_index, 
            dtype=torch.long, 
            device=device
        )
        padded_attention_mask = torch.zeros(
            batch_size, max_seq_len, 
            dtype=torch.long, 
            device=device
        )
        
        # Fill in the padded tensors
        for i in range(batch_size):
            seq_len = min(batch_embeddings[i].shape[0], max_seq_len)
            
            if seq_len == 0:
                continue
            
            if self.padding_side == "right":
                # Content at the beginning, padding at the end
                padded_embeddings[i, :seq_len] = batch_embeddings[i][:seq_len]
                padded_labels[i, :seq_len] = batch_labels[i][:seq_len]
                padded_attention_mask[i, :seq_len] = batch_attention_masks[i][:seq_len]
            else:
                # Padding at the beginning, content at the end (left padding)
                padded_embeddings[i, -seq_len:] = batch_embeddings[i][:seq_len]
                padded_labels[i, -seq_len:] = batch_labels[i][:seq_len]
                padded_attention_mask[i, -seq_len:] = batch_attention_masks[i][:seq_len]
        
        return {
            'inputs_embeds': padded_embeddings,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
        }
    
    def __repr__(self) -> str:
        status = []
        if self._inputs is not None:
            status.append(f"inputs={len(self._inputs)} samples")
        if self._labels is not None:
            status.append(f"labels={len(self._labels)} samples")
        if self._transactions is not None:
            status.append(f"transactions={list(self._transactions.keys())}")
        
        status_str = ", ".join(status) if status else "empty"
        return f"MultiModalBatchProcessor({status_str})"
```

## Usage Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Initialize processor
processor = MultiModalBatchProcessor(
    tokenizer=tokenizer,
    system_prompt="You are a financial analysis assistant.",
    max_length=2048,
    padding_side="right",
)

# Example batch data
batch_size = 2
emb_dim = model.config.hidden_size  # e.g., 4096

# User inputs with placeholders for transaction embeddings
inputs = [
    ["Analyze the following transaction:", "<trans_1>", "What pattern do you see?"],
    ["Compare these two transactions:", "<trans_2>", "and", "<trans_3>", "Are they related?"],
]

# Target responses (labels)
labels = [
    "This transaction shows a typical retail purchase pattern with moderate amount.",
    "These transactions appear to be related transfers between linked accounts.",
]

# Transaction embeddings (from your transaction encoder)
transactions = {
    "<trans_1>": torch.randn(batch_size, emb_dim),
    "<trans_2>": torch.randn(batch_size, emb_dim),
    "<trans_3>": torch.randn(batch_size, emb_dim),
}

# Process the batch
processor.process_inputs(inputs, labels)
processor.add_transactions(transactions)

# Resolve to final format
result = processor.resolve_batch(
    embed_layer=model.get_input_embeddings(),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Use with model
outputs = model(
    inputs_embeds=result['inputs_embeds'],
    attention_mask=result['attention_mask'],
    labels=result['labels'],
)

print(f"Loss: {outputs.loss.item()}")

# Clean up for next batch
processor.reset()
