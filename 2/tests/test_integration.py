import torch
import unittest
from attention import apply_rotary_pos_emb, rotate_half

class TestIntegration(unittest.TestCase):
    def test_apply_rotary_pos_emb_with_rotate_half(self):
        batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 8
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        cos = torch.cos(torch.arange(seq_len).float()).unsqueeze(-1).expand(-1, head_dim).unsqueeze(0).expand(batch_size, -1, -1)
        sin = torch.sin(torch.arange(seq_len).float()).unsqueeze(-1).expand(-1, head_dim).unsqueeze(0).expand(batch_size, -1, -1)

        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
        # Ensure rotated values incorporate rotate_half correctly
        self.assertTrue(torch.allclose(q_rot, q * cos + rotate_half(q) * sin, atol=1e-5))
        self.assertTrue(torch.allclose(k_rot, k * cos + rotate_half(k) * sin, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
