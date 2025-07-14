import torch.nn as nn
import math
import unittest
import torch

class TokenEmbedding(nn.Module):
    """This embedding layer will be used to map token
       to corresponding d_model size vector.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class TestTokenEmbedding(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.vocab_size = 50
        self.embedding = TokenEmbedding(self.d_model, self.vocab_size)

    def test_output_shape(self):
        # Test with a batch of token indices
        x = torch.randint(0, self.vocab_size, (4, 10))  # batch_size=4, seq_len=10
        out = self.embedding(x)
        self.assertEqual(out.shape, (4, 10, self.d_model))


if __name__ == "__main__":
    unittest.main()