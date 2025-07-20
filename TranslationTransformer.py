import unittest
import torch
from PositionalEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
import torch.nn as nn
from tqdm import tqdm

###########################################################
# Define wrapper class for Transformer, and tests

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
    
    def get_tgt_mask(self, tgt_len: int) -> torch.tensor:
        """
            Generte a mask to prevent to ensures that during training,
            the model cannot “see” future tokens in the target sequence,
            simulating the real inference stage where the model only has
            access to previously generated tokens.

            Arg:
                tgt_len: Target 
        """
        return nn.Transformer.generate_square_subsequent_mask(tgt_len - 1).bool()  # -1 for decoder input


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

    def test_get_tgt_mask(self):
        # Test that get_tgt_mask returns a square mask of correct shape and type
        tgt_len = 10
        mask = self.model.get_tgt_mask(tgt_len)
        self.assertEqual(mask.shape, (tgt_len - 1, tgt_len - 1))
        self.assertTrue(mask.dtype == torch.bool)
        # Check that the mask is upper triangular (future tokens masked)
        self.assertTrue(torch.equal(mask, torch.triu(torch.ones_like(mask), diagonal=1).bool()))

if __name__ == "__main__":
    unittest.main()
###########################################################


###########################################################
# Define training loop and perdict funtions 
# that utilize the Transformer

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(model, optimizer, criterion, train_dataloader):
    """
    Runs one epoch of training for the translation transformer model.

    Args:
        model: The TranslationTransformer instance.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        train_dataloader: DataLoader yielding batches of training data.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    train_loss = 0

    for batch in tqdm(train_dataloader):
        # Move batch tensors to the correct device
        src_tokens = batch["src_tokens"].to(device)
        dec_tokens = batch["dec_tokens"].to(device)
        label_tokens = batch["label_tokens"].to(device)
        tgt_padding_mask = batch["tgt_padding_mask"].to(device)
        src_padding_mask = batch["src_padding_mask"].to(device)
        # Generate target mask for decoder to prevent attending to future tokens
        tgt_mask = model.get_tgt_mask(train_dataloader.tgt_len).to(device)

        optimizer.zero_grad()
        # Forward pass through the model
        logits = model(src_tokens, dec_tokens, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), label_tokens.view(-1))
        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_dataloader)

def validation_loop(model, criterion, test_dataloader):
    """
    Runs one epoch of validation for the translation transformer model.

    Args:
        model: The TranslationTransformer instance.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        test_dataloader: DataLoader yielding batches of validation data.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # Move batch tensors to the correct device
            src_tokens = batch["src_tokens"].to(device)
            dec_tokens = batch["dec_tokens"].to(device)
            label_tokens = batch["label_tokens"].to(device)
            tgt_padding_mask = batch["tgt_padding_mask"].to(device)
            src_padding_mask = batch["src_padding_mask"].to(device)
            # Generate target mask for decoder to prevent attending to future tokens
            tgt_mask = model.get_tgt_mask(test_dataloader.tgt_len).to(device)

            # Forward pass through the model
            logits = model(src_tokens, dec_tokens, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), label_tokens.view(-1))
            test_loss += loss.item()

    return test_loss / len(test_dataloader)

def fit(model, optimizer, criterion, train_dataloader, test_dataloader, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_loop(model, optimizer, criterion, train_dataloader)
        validation_loss = validation_loop(model, criterion, test_dataloader)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}\n")
###########################################################