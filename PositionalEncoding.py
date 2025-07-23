import unittest
import torch
from torch import nn,Tensor
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()

    # initialize dropout
    self.dropout = nn.Dropout(p=dropout)

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)

    # create position column
    k = torch.arange(0, max_length).unsqueeze(1)

    # calc divisor for positional encoding
    div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)

    # calc cosine on odd indices
    pe[:, 1::2] = torch.cos(k * div_term)

    # add dimension
    pe = pe.unsqueeze(0)

    # buffers are saved in state_dict but not trained by the optimizer
    self.register_buffer("pe", pe)

  def forward(self, x: Tensor):
    """
    Args:
      x:  embeddings (batch_size, seq_length, d_model)

    Returns:
          embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False)

    # perform dropout
    return self.dropout(x)

class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.max_seq_length = 32
        self.dropout = 0.1
        self.pe = PositionalEncoding(self.d_model, self.max_seq_length, self.dropout)

    def test_output_shape(self):
        # Input shape: (batch_size, seq_len, d_model)
        x = torch.zeros(4, 10, self.d_model)
        out = self.pe(x)
        self.assertEqual(out.shape, (4, 10, self.d_model))



if __name__ == "__main__":
    unittest.main()