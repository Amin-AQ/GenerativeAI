import torch
import unittest
from layers import RMSNorm

class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 16
        self.batch_size = 4
        self.seq_length = 8
        self.eps = 1e-6
        self.norm_layer = RMSNorm(self.hidden_size, eps=self.eps)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        output = self.norm_layer(x)
        self.assertEqual(output.shape, x.shape, "Output shape mismatch!")

    def test_normalization_effect(self):
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        output = self.norm_layer(x)
        rms_x = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        expected = x / rms_x * self.norm_layer.weight
        self.assertTrue(torch.allclose(output, expected, atol=1e-5), "RMSNorm not applied correctly!")

    def test_learnable_weight(self):
        x = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        self.norm_layer.weight.data.fill_(2.0)  # Set weights to 2
        output = self.norm_layer(x)
        rms_x = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        expected = (x / rms_x) * 2.0
        self.assertTrue(torch.allclose(output, expected, atol=1e-5), "Learnable scaling incorrect!")

if __name__ == "__main__":
    unittest.main()
