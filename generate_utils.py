import re
from typing import Sequence

def _parse_response(texts: Sequence[str], patterns: Sequence[str]) -> list[dict[str, int]]:
    """
    Parse text strings to extract numeric values associated with patterns.
    
    Args:
        texts: Sequence of text strings to parse
        patterns: Sequence of pattern strings to search for
        
    Returns:
        List of dictionaries mapping pattern names to extracted integers
    """
    results = []
    
    for text in texts:
        result_dict = {}
        
        for pattern in patterns:
            # Create regex pattern that matches:
            # - the pattern word (case-insensitive, allowing slight variations)
            # - optional separator characters (: , ; space, etc.)
            # - a number (integer)
            regex_pattern = rf'\b{re.escape(pattern)}\b\s*[:;,]?\s*(-?\d+)'
            
            match = re.search(regex_pattern, text, re.IGNORECASE)
            
            if match:
                result_dict[pattern] = int(match.group(1))
            else:
                result_dict[pattern] = None
        
        results.append(result_dict)
    
    return results


from typing import List
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

def _generate_response(inputs: dict[str, Tensor]) -> List[Tensor]:
    """
    Prepare batch generation input by selecting embeddings where labels == -100 
    and attention_mask == 1, then apply left padding for generation.
    
    Args:
        inputs: Dict containing 'inputs_embeds', 'attention_mask', 'labels'
        
    Returns:
        List of generated token tensors from model.generate()
    """
    inputs_embeds = inputs['inputs_embeds']  # (batch_size, seq_len, hidden_dim)
    attention_mask = inputs['attention_mask']  # (batch_size, seq_len)
    labels = inputs['labels']  # (batch_size, seq_len)
    
    batch_size = inputs_embeds.shape[0]
    
    # Step 1: Select embeddings where labels == -100 AND attention_mask == 1
    valid_mask = (labels == -100) & (attention_mask == 1)
    
    # Extract valid embeddings for each sample
    valid_embeds_list = []
    for i in range(batch_size):
        sample_embeds = inputs_embeds[i][valid_mask[i]]
        valid_embeds_list.append(sample_embeds)
    
    # Step 2: Apply left padding using flip + pad_sequence + flip trick
    # Flip sequences, apply right padding, then flip back for left padding
    flipped_embeds = [torch.flip(embeds, dims=[0]) for embeds in valid_embeds_list]
    padded_flipped = pad_sequence(flipped_embeds, batch_first=True, padding_value=0.0)
    padded_embeds = torch.flip(padded_flipped, dims=[1])
    
    # Create generation attention mask (1 for valid tokens, 0 for padding)
    gen_attention_mask = torch.zeros(
        batch_size, padded_embeds.shape[1],
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )
    for i, embeds in enumerate(valid_embeds_list):
        valid_len = len(embeds)
        gen_attention_mask[i, -valid_len:] = 1  # Non-padded positions on the right
    
    # Step 3: Call generate
    generated_ids = model.generate(
        inputs_embeds=padded_embeds,
        attention_mask=gen_attention_mask,
        max_new_tokens=50,  # Adjust as needed
        # Add other generation parameters
    )
    
    return generated_ids
