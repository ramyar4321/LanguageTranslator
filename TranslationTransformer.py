import unittest
import torch
from PositionalEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
import torch.nn as nn
from tqdm import tqdm
import re

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

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """Encodes the source sequence using the transformer encoder.
        Args:
            src (torch.Tensor): Source token indices (batch_size, src_seq_len).
            src_mask (torch.Tensor): Source attention mask.
        Returns:
            torch.Tensor: Encoded memory (batch_size, src_seq_len, d_model).
        """
        return self.transformer.encoder(self.pos_encoder(
                            self.src_embedding(src)), src_mask)
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        """Decodes the target sequence using the transformer decoder.
        Args:  
            tgt (torch.Tensor): Target token indices (batch_size, tgt_seq_len).
            memory (torch.Tensor): Encoded memory from the encoder (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor): Target attention mask.
            Returns:
                torch.Tensor: Decoded output (batch_size, tgt_seq_len, d_model).
        """
        return self.transformer.decoder(self.pos_encoder(
                          self.tgt_embedding(tgt)), memory,
                          tgt_mask)
    
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

    def test_encode_output_shape(self):
        # Test that encode returns correct shape
        batch_size = 3
        src_seq_len = 7
        src = torch.randint(0, self.src_vocab_size, (batch_size, src_seq_len))
        src_mask = torch.zeros(src_seq_len, src_seq_len)
        # Should return (batch_size, src_seq_len, d_model)
        memory = self.model.encode(src, src_mask)
        self.assertEqual(memory.shape, (batch_size, src_seq_len, self.d_model))

    def test_decode_output_shape(self):
        # Test that decode returns correct shape
        batch_size = 2
        tgt_seq_len = 5
        src_seq_len = 8
        tgt = torch.randint(0, self.tgt_vocab_size, (batch_size, tgt_seq_len))
        memory = torch.randn(batch_size, src_seq_len, self.d_model)
        tgt_mask = torch.zeros(tgt_seq_len, tgt_seq_len)
        # Should return (batch_size, tgt_seq_len, d_model)
        out = self.model.decode(tgt, memory, tgt_mask)
        self.assertEqual(out.shape, (batch_size, tgt_seq_len, self.d_model))

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

def translate(eng_sentence: str, model: TranslationTransformer, 
            tokenizer_en, tokenizer_fr, max_tgt_len):
    """
    Translate an English sentence to French using the trained Transformer model.
    
    Args:
        eng_sentence (str): Input English sentence
        model (nn.Transformer): Trained Transformer model
        tokenizer_en (Tokenizer): English tokenizer
        tokenizer_fr (Tokenizer): French tokenizer
        max_len (int): Maximum length of the generated French sentence
    
    Returns:
        str: Translated French sentence
    """
    
    # Preprocess the input English sentence
    eng_sentence = eng_sentence.lower()
    eng_sentence = re.sub(r'[^a-zA-ZÀ-ÿ!? \.]', '', eng_sentence)
    
    # Tokenize and prepare source input
    enc_input_tokens = tokenizer_en.encode(eng_sentence).ids
    src_tokens = torch.cat([
        torch.tensor([tokenizer_en.token_to_id("[SOS]")], dtype=torch.int64),
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        torch.tensor([tokenizer_en.token_to_id("[EOS]")], dtype=torch.int64),
        torch.tensor([tokenizer_en.token_to_id("[PAD]")], dtype=torch.int64).repeat(max_en_len - len(enc_input_tokens) - 2)
    ]).unsqueeze(0).to(device) # Shape: [1, src_len]


    # Initialize target sequence with [SOS]
    tgt_tokens = torch.tensor([tokenizer_fr.token_to_id("[SOS]")], dtype=torch.int64).unsqueeze(0).to(device)  # Shape: [1, 1]
    
    # Autoregressive decoding
    tgt_tokens=perdict(model, src_tokens, tgt_tokens, max_tgt_len,tokenizer_fr.token_to_id("[EOS]"))
    
    # Decode the token sequence to a French sentence
    fr_ids = tgt_tokens[0].cpu().tolist()
    fr_sentence = tokenizer_fr.decode(fr_ids)
    
    # Clean up the output (remove special tokens)
    fr_sentence = fr_sentence.replace("[SOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
    
    return fr_sentence

def perdict(model: TranslationTransformer, 
            src_tokens, tgt_tokens, max_tgt_len, EOS_IDX):
    """
        
    """
    model.eval()
    
    
    # Encode the source sentence
    memory = model.encoder(src_tokens)  # Shape: [1, src_len, d_model]

    # Autoregressive decoding
    for i in range(max_tgt_len):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).bool().to(device)
        
        # Decode step
        output = model.decoder(tgt_tokens, memory, tgt_mask=tgt_mask) # Shape: [1, tgt_len, d_model] 
        logits = model.generator(output[:, -1, :])  # Predict next token: [1, fr_vocab]
        next_token = torch.argmax(logits, dim=-1)  # Shape: [1]
        
        # Append predicted token
        tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(0)], dim=1)  # Shape: [1, tgt_len + 1]
        
        # Stop if [EOS] is predicted
        if next_token.item() == EOS_IDX:
            break
    
    
    return tgt_tokens


###########################################################