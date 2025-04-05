import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

def rotate_half(x: torch.Tensor):
    """
    Rotates the left half of a tensor along its final dimension.

    This function is used in Rotary Positional Embeddings (RoPE) to apply a 
    complex-valued rotation by swapping the two halves of the last dimension
    with a sign flip.

    Given an input tensor `x` of shape (..., head_dim), it splits `x` into 
    two equal halves along the last dimension, then swaps them while negating 
    the second half.

    Args:
        x (torch.Tensor): Input tensor of shape (..., head_dim), where head_dim must be even.

    Returns:
        torch.Tensor: The rotated tensor of the same shape as `x`, where the two halves 
                      are swapped with sign inversion on the second half.
    """
    # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
    # WRITE YOUR CODE HERE 
    head_dim = x.size(dim=-1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, but got {head_dim}")
    x_left, x_right = x[..., :head_dim // 2], -x[..., head_dim // 2:]

    # swaping and concatentaion
    x_concatenated = torch.cat((x_right, x_left),dim=-1)
    return x_concatenated

def apply_rotary_pos_emb(q, k, cos: torch.Tensor, sin: torch.Tensor, position_ids=None, unsqueeze_dim=None):
    """
    Applies Rotary Positional Embeddings (RoPE) to the query and key tensors.

    Rotary Positional Embeddings (RoPE) encode positional information directly 
    into the query and key representations by rotating them in a complex-valued 
    space. This method enhances the model's ability to capture relative positions.

    Args:
        q (torch.Tensor): Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k (torch.Tensor): Key tensor of shape (batch, num_heads, seq_len, head_dim).
        cos (torch.Tensor): Precomputed cosine values for RoPE of shape (seq_len, head_dim).
        sin (torch.Tensor): Precomputed sine values for RoPE of shape (seq_len, head_dim).
        position_ids (torch.Tensor, optional): Position indices of shape (batch, seq_len).
                                               Defaults to None, which assumes sequential positions.
        unsqueeze_dim (int, optional): If provided, expands `cos` and `sin` along this dimension
                                       to facilitate broadcasting.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors with the same shape
                                           as the input (batch, num_heads, seq_len, head_dim).
    """
    # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
    # WRITE CODE HERE

    # need to unsqueeze cos and sine if needed
    if unsqueeze_dim is not None:
        sin = sin.unsqueeze(unsqueeze_dim)
        cos = cos.unsqueeze(unsqueeze_dim)
    
    half_rotated_q, half_rotated_k = rotate_half(q), rotate_half(k)
    # print(cos.shape, sin.shape)
    # print(q.shape)
    q_rotated = q * cos + half_rotated_q * sin
    k_rotated = k * cos + half_rotated_k * sin
    return (q_rotated, k_rotated)

class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        # Precompute frequency for sine/cosine embeddings
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # shape = head_dim//2

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # WRITE CODE HERE 
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_ids_expanded = position_ids[:,None,:]  # batch, 1, seq

        freq_expanded = self.freq[None, :, None]   # 1, head//2, 1
        freq_expanded = freq_expanded.expand(pos_ids_expanded.shape[0],-1,1)  # batch, head//2, 1

        angles = freq_expanded.float() @ pos_ids_expanded.float()   # batch, head//2, seq 
        angles = angles.transpose(1,2)  # batch, seq, head//2 
        embeddings = torch.cat((angles,angles), dim=-1)    # batch_size, seq_len, head_dim

        return embeddings.sin(), embeddings.cos()

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        # Model dimensions and attention configurations
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads  # Number of key-value heads
        self.rope_theta = 10000.0  # Scaling factor for rotary embeddings

        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary embedding generator
        self.rotary_emb = RotaryEmbedder(base=self.rope_theta, dim=self.head_dim)
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int):
        """
        Expands the number of key-value attention heads by repeating them.

        This function is used in grouped query attention (GQA) and multi-query attention (MQA) 
        to duplicate key-value heads `n_rep` times, so they can be shared across multiple query heads.

        Args:
            x (torch.Tensor): A tensor of shape (batch, num_key_value_heads, seq_len, head_dim),
                            representing the key or value tensor.
            n_rep (int): The number of times to repeat each key-value head.

        Returns:
            torch.Tensor: A tensor of shape (batch, num_key_value_heads * n_rep, seq_len, head_dim),
                        where each key-value head is repeated `n_rep` times.
        """
        # WRITE CODE HERE 
        batch, num_key_value_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)  # (batch, key_value_heads, 1, seq_len, head_dim)
        x = x.repeat(1, 1, n_rep, 1, 1)  # (batch, key_value_heads, n_rep, seq_len, head_dim)
        x = x.view(batch, num_key_value_heads * n_rep, seq_len, head_dim)  # (batch, num_heads, seq_len, head_dim)
        return x
        


    def forward(self, x: torch.Tensor, attention_mask=None):
        # WRITE YOUR CODE HERE
        # x = batch, seq_len, emb_dim
        batch_size, seq_len, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        hidden_kv_shape = (batch_size, seq_len, self.kv_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(*hidden_kv_shape).transpose(1,2)
        v = v.view(*hidden_kv_shape).transpose(1,2)        # batch_size, heads, seq_len, head_dim

        sin, cos = self.rotary_emb(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, None, unsqueeze_dim=1)  # batch_size, heads, seq_len, head_dim

        n_rep = self.num_heads//self.kv_heads
        keys = self._repeat_kv(k, n_rep)        # batch, num_key_value_heads * n_rep, seq_len, head_dim
        values = self._repeat_kv(v, n_rep)

        scores = torch.matmul(q, keys.transpose(2,3)) / math.sqrt(self.head_dim)  # batch, heads, seq_len, seq_len

        if attention_mask is not None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            tokenizer_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq)
            attention_mask = causal_mask * tokenizer_mask  # (batch, 1, seq, seq)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        scores = F.softmax(scores.float(), dim=-1)

        H = torch.matmul(scores, values)   #  batch, heads, seq_len, head_dim

        o = self.o_proj(H.transpose(1,2).contiguous().view(batch_size,seq_len,-1))  # batch, seq_len, emb_dim same shape as input to forward func
        
        return o