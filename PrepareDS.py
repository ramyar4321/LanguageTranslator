import torch
import torch.nn as nn
from torch.utils.data import Dataset
from BPE import BPE
import unittest

class PrepareDS(Dataset):
    # Prepares the dataset to be feed into a Transformer 
    def __init__(self, bpe_src, bpe_tgt, src_text, tgt_text, src_len, tgt_len):
        self.tokenizer_src = bpe_src
        self.tokenizer_tgt = bpe_tgt
        self.src = src_text
        self.tgt = tgt_text
        self.src_len = src_len 
        self.tgt_len = tgt_len 

        self.sos_token = torch.tensor([self.tokenizer_src.special_tokens["SOS"]], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_src.special_tokens["EOS"]], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_src.special_tokens["PAD"]], dtype=torch.int64)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]

        enc_input_tokens = self.tokenizer_src.tokenize(src_text)
        dec_input_tokens = self.tokenizer_tgt.tokenize(tgt_text)

        enc_padding = self.src_len - len(enc_input_tokens) - 2  # -2 for SOS/EOS
        dec_padding = self.tgt_len - len(dec_input_tokens) - 2  # -2 for SOS/EOS

        # Ensure padding is non-negative
        enc_padding = max(0, enc_padding)
        dec_padding = max(0, dec_padding)

        # Uncomment if using set sequence length instead of max
        # seuqence length
        ## Pad sequences to the maximum lengths if they are shorter
        ## If sequences are longer, truncate them
        #enc_input_tokens = enc_input_tokens[:self.src_len - 2] # -2 for SOS/EOS
        #dec_input_tokens = dec_input_tokens[:self.tgt_len - 2]

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(enc_padding)
        ])

        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(dec_padding)
        ])

        return {
            "src_tokens": encoder_input,
            "dec_tokens": dec_input[:-1],  # Decoder input: [SOS] + tokens
            "label_tokens": dec_input[1:],  # Target: tokens + [EOS]
            "tgt_padding_mask": (dec_input[:-1] == self.pad_token).bool(),
            "src_padding_mask": (encoder_input == self.pad_token).bool(),
        }

class TestPrepareDS(unittest.TestCase):
    def setUp(self):
        # Use a small vocab size for testing
        self.src_texts = ["Hello, how are you?", 
                          "I would like a cup of coffee.",
                          "The weather is nice today.",
                          "Can you help me, please?",
                          "I am learning French.",]
        self.tgt_texts = ["Bonjour, comment ça va ?",
                          "Je voudrais une tasse de café.",
                          "Il fait beau aujourd'hui.",
                          "Pouvez-vous m'aider, s'il vous plaît ?",
                          "J'apprends le français."]
        self.src_len = 40
        self.tgt_len = 40
        self.bpe_src = BPE(3)
        self.bpe_tgt = BPE(3)
        # Train BPEs on the test data to initialize vocab and merges
        self.bpe_src.train(" ".join(self.src_texts))
        self.bpe_tgt.train(" ".join(self.tgt_texts))
        self.ds = PrepareDS(
            self.bpe_src, self.bpe_tgt,
            self.src_texts, self.tgt_texts,
            self.src_len, self.tgt_len
        )

    def test_len(self):
        self.assertEqual(len(self.ds), 5)

    def test_getitem_shapes(self):
        item = self.ds[0]
        self.assertEqual(item["src_tokens"].shape[0], self.src_len)
        self.assertEqual(item["dec_tokens"].shape[0], self.tgt_len - 1)
        self.assertEqual(item["label_tokens"].shape[0], self.tgt_len - 1)
        self.assertEqual(item["src_padding_mask"].shape[0], self.src_len)
        self.assertEqual(item["tgt_padding_mask"].shape[0], self.tgt_len - 1)

    def test_special_tokens(self):
        item = self.ds[0]
        # SOS token should be first
        self.assertEqual(item["src_tokens"][0].item(), self.bpe_src.special_tokens["SOS"])
        self.assertEqual(item["dec_tokens"][0].item(), self.bpe_tgt.special_tokens["SOS"])
        # EOS token should be present
        self.assertIn(self.bpe_src.special_tokens["EOS"], item["src_tokens"])
        self.assertIn(self.bpe_tgt.special_tokens["EOS"], item["dec_tokens"])
        # PAD token should be present if padding is needed
        self.assertIn(self.bpe_src.special_tokens["PAD"], item["src_tokens"])

    def test_padding_mask(self):
        item = self.ds[0]
        # Padding mask should be True where PAD token is present
        pad_indices = (item["src_tokens"] == self.bpe_src.special_tokens["PAD"]).nonzero(as_tuple=True)[0]
        for idx in pad_indices:
            self.assertTrue(item["src_padding_mask"][idx])

if __name__ == "__main__":
    unittest.main()