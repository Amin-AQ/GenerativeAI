import torch
import unittest
from attention import rotate_half, apply_rotary_pos_emb

class TestRotateHalf(unittest.TestCase):
    def test_rotate_half_functionality(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0], [3.0, 4.0, -1.0, -2.0]])
        output = rotate_half(x)
        self.assertTrue(torch.equal(output, expected), "rotate_half output incorrect!")

    def test_invalid_head_dim(self):
        x = torch.randn(2, 3)  # Last dimension (head_dim) is odd
        with self.assertRaises(ValueError):
            rotate_half(x)

class TestApplyRotaryPosEmb(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 6
        self.head_dim = 8  # Must be even
        
        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        
        # Precomputed cos and sin embeddings
        self.cos = torch.cos(torch.arange(self.seq_len).float()).unsqueeze(-1).expand(-1, self.head_dim).unsqueeze(0).expand(self.batch_size, -1, -1)
        self.sin = torch.sin(torch.arange(self.seq_len).float()).unsqueeze(-1).expand(-1, self.head_dim).unsqueeze(0).expand(self.batch_size, -1, -1)


    def test_output_shape(self):
        q_rot, k_rot = apply_rotary_pos_emb(self.q, self.k, self.cos, self.sin, unsqueeze_dim=1)
        self.assertEqual(q_rot.shape, self.q.shape, "Rotary Q shape mismatch!")
        self.assertEqual(k_rot.shape, self.k.shape, "Rotary K shape mismatch!")

    def test_rotation_properties(self):
        q_rot, k_rot = apply_rotary_pos_emb(self.q, self.k, self.cos, self.sin, unsqueeze_dim=1)
        sin = self.sin.unsqueeze(1)
        cos = self.cos.unsqueeze(1)
        # Check if rotation is applied correctly by verifying transformation consistency
        q_expected = self.q * cos + rotate_half(self.q) * sin
        k_expected = self.k * cos + rotate_half(self.k) * sin

        self.assertTrue(torch.allclose(q_rot, q_expected, atol=1e-5), "Q rotation incorrect!")
        self.assertTrue(torch.allclose(k_rot, k_expected, atol=1e-5), "K rotation incorrect!")

    def test_unsqueeze_dimension(self):
        q_rot, k_rot = apply_rotary_pos_emb(self.q, self.k, self.cos, self.sin, unsqueeze_dim=1)
        self.assertEqual(q_rot.shape, self.q.shape, "Unsqueeze behavior changed shape!")

if __name__ == "__main__":
    unittest.main()
