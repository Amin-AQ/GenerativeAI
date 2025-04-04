import torch
import unittest
from attention import RotaryEmbedder  # Assuming your class is in `rotary_embedder.py`

class TestRotaryEmbedder(unittest.TestCase):

    def setUp(self):
        """Initialize common test parameters."""
        self.batch_size = 4
        self.num_heads = 8
        self.seq_len = 16
        self.head_dim = 64
        self.base = 10000  # Standard RoPE base
        self.embedder = RotaryEmbedder(dim=self.head_dim, base=self.base)

    def test_output_shape(self):
        """Test that the output has the correct shape."""
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        sin_emb, cos_emb = self.embedder(x)

        # Expected shape: (batch_size, seq_len, head_dim)
        expected_shape = (self.batch_size, self.seq_len, self.head_dim)
        self.assertEqual(sin_emb.shape, expected_shape)
        self.assertEqual(cos_emb.shape, expected_shape)

    def test_value_range(self):
        """Ensure outputs are within valid sin/cos value ranges (-1 to 1)."""
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        sin_emb, cos_emb = self.embedder(x)

        self.assertTrue(torch.all(sin_emb >= -1) and torch.all(sin_emb <= 1))
        self.assertTrue(torch.all(cos_emb >= -1) and torch.all(cos_emb <= 1))

    def test_single_sequence_length(self):
        """Test with a sequence length of 1."""
        x = torch.randn(self.batch_size, self.num_heads, 1, self.head_dim)
        sin_emb, cos_emb = self.embedder(x)

        expected_shape = (self.batch_size, 1, self.head_dim)
        self.assertEqual(sin_emb.shape, expected_shape)
        self.assertEqual(cos_emb.shape, expected_shape)

    def test_large_dim(self):
        """Test with a large head_dim."""
        large_dim = 128
        embedder = RotaryEmbedder(dim=large_dim, base=self.base)
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, large_dim)
        sin_emb, cos_emb = embedder(x)

        expected_shape = (self.batch_size, self.seq_len, large_dim)
        self.assertEqual(sin_emb.shape, expected_shape)
        self.assertEqual(cos_emb.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()
