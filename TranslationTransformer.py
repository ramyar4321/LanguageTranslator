import unittest
import torch
from PositionalEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
import torch.nn as nn


class TranslationTransformer(nn.Module):
    """
    A Transformer model for sequence-to-sequence translation tasks.
    This class wraps PyTorch's nn.Transformer and provides embedding layers
    for source and target vocabularies, positional encoding, and a generator
    for output token logits.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        d_model (int): Dimension of the embeddings and transformer hidden states.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        max_seq_length (int): Maximum sequence length 
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        # Define layers
        self.src_embedding = TokenEmbedding(d_model ,src_vocab_size)
        self.tgt_embedding = TokenEmbedding(d_model, tgt_vocab_size)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        # Final linear layer to project transformer output to target vocabulary logits
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass for the transformer.

        Args:
            src (Tensor): Source token indices (batch_size, src_seq_len).
            tgt (Tensor): Target token indices (batch_size, tgt_seq_len).
            src_mask (Tensor, optional): Source attention mask.
            tgt_mask (Tensor, optional): Target attention mask.
            src_key_padding_mask (Tensor, optional): Source padding mask.
            tgt_key_padding_mask (Tensor, optional): Target padding mask.

        Returns:
            Tensor: Output logits for target vocabulary (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # Embed and apply positional encoding to source and target tokens
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        # Pass through transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # Project transformer output to vocabulary logits
        return self.generator(output)


class TestTranslationTransformer(unittest.TestCase):
    def setUp(self):
        self.src_vocab_size = 50
        self.tgt_vocab_size = 60
        self.d_model = 32
        self.max_seq_length = 16
        self.model = TranslationTransformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=self.d_model,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            dropout=0.1,
            max_seq_length=self.max_seq_length
        )

    def test_forward_output_shape(self):
        batch_size = 4
        src_seq_len = 10
        tgt_seq_len = 12
        src = torch.randint(0, self.src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (batch_size, tgt_seq_len))
        out = self.model(src, tgt)
        self.assertEqual(out.shape, (batch_size, tgt_seq_len, self.tgt_vocab_size))

    def test_forward_with_masks(self):
        batch_size = 2
        src_seq_len = 8
        tgt_seq_len = 7
        src = torch.randint(0, self.src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (batch_size, tgt_seq_len))
        src_mask = torch.zeros(src_seq_len, src_seq_len)
        tgt_mask = torch.zeros(tgt_seq_len, tgt_seq_len)
        src_key_padding_mask = torch.zeros(batch_size, src_seq_len, dtype=torch.bool)
        tgt_key_padding_mask = torch.zeros(batch_size, tgt_seq_len, dtype=torch.bool)
        out = self.model(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        self.assertEqual(out.shape, (batch_size, tgt_seq_len, self.tgt_vocab_size))

    def test_gradients(self):
        batch_size = 2
        src_seq_len = 5
        tgt_seq_len = 6
        src = torch.randint(0, self.src_vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, self.tgt_vocab_size, (batch_size, tgt_seq_len))
        out = self.model(src, tgt)
        loss = out.sum()
        loss.backward()
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

if __name__ == "__main__":
    unittest.main()