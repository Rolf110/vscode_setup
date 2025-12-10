import re
from typing import Sequence

def _parse_response(texts: Sequence[str], patterns: Sequence[str]) -> list[dict[str, list[int] | None]]:
    """
    Parse text sequences to extract numbers associated with specific patterns.
    
    Args:
        texts: Sequence of strings to parse
        patterns: Sequence of pattern strings to search for
        
    Returns:
        List of dictionaries, one per text, mapping each pattern to:
        - list of integers if pattern found with numbers
        - None if pattern not found
    """
    results = []
    
    for text in texts:
        text_result = {}
        
        for pattern in patterns:
            # Create regex to find pattern followed by optional separator and number
            # Pattern: exact pattern match + optional `:` or space + capture digits
            regex = rf'\b{re.escape(pattern)}\s*:?\s*(\d+)'
            
            # Find all occurrences
            matches = re.findall(regex, text)
            
            if matches:
                # Convert string matches to integers
                text_result[pattern] = [int(m) for m in matches]
            else:
                # Pattern not found in text
                text_result[pattern] = None
        
        results.append(text_result)
    
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
