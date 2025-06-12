import torch
from torch import Tensor
import unittest
import os
from IndexedList import IndexedList
from CountPair import CountPair
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

class GPU_BPE:

    def __init__(self):
        self.merges = []
        self.special_tokens = {}
        self.vocab = {}

    def build_indexed_list(self, text: str) -> IndexedList:  
        # Create an IndexedList with the encoded bytes.
        return IndexedList(t for t in text.encode('utf-8'))

    def cpu_merge(self, pair: tuple[str, str], new_id: int, 
                indexed_list: IndexedList, stats: CountPair=None) -> None:
        """ 
            Find all instances of the given pair in the index list 
            and merge them to form the new_id. If this function
            is called by the train function, then the CountPair is
            updated for each merge.

            This function performs M merges in the IndexedList
            where each merge is O(1). For each merge, CountPair
            is updated which is also O(log L) where L is the training text
            length, thus this function has a time complexity of O(M LogL)
            
        """
        for node in indexed_list.index[pair]: # O(M)
            if node.val != pair[0] or node.next is None or node.next.val != pair[1]:
                # The index was stale - continue.
                continue
            # Say we're merging "bc" to "X" in "abcd", 
            # and the node we're visiting now is "b".
            # O(log L) due to heapify operations
            if stats is not None:  # Update the stats.
                stats.remove(pair)  # Remove "bc".
                if node.next.next is not None:
                    stats.remove((node.next.val, node.next.next.val))  # Remove "cd".
                    stats.add((new_id, node.next.next.val))  # Add "Xd".
                if node.prev is not None:
                    stats.remove((node.prev.val, pair[0]))  # Remove "ab".
                    stats.add((node.prev.val, new_id))  # Add "aX".
            # O(1)
            node.next.delete()  # Delete "c", we now have "abd".
            node.val = new_id  # Update "b" to "X", we now have "aXd".
            indexed_list.update_index(node)  # Add "aX" and "Xd" to the index.


    def merge(self, ids: Tensor, pair: Tensor, idx: int):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """

        # create a mask for the first element i of every matching pair (i, j)
        pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
        is_pair = (pairs == pair).all(axis=1)
        false_tensor = torch.tensor([False], dtype=torch.bool, device=ids.device)
        is_pair_i = torch.cat((is_pair, false_tensor))

        # create a mask for the second element j of every matching pair (i, j)
        is_pair_j = is_pair_i.roll(1)

        # handle overlapping pairs for repeated tokens
        while True:
            is_overlap = (is_pair_i & is_pair_j).any()
            if not is_overlap:
                break # no overlapping pairs

            # remove first overlapping pairs in repeated sequences
            is_first = (is_pair_i & is_pair_j).int().diff() == 1
            is_first = torch.cat((false_tensor, is_first))
            is_pair_i &= ~is_first
            is_pair_j = is_pair_i.roll(1)

        # change the first element i of every matching pair (i, j) to the new token
        ids[is_pair_i] = idx

        # remove the second element j of every matching pair (i, j)
        ids = ids[~is_pair_j]
        return ids

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = []
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)

        for i in range(num_merges):
            # determine the most common pair to merge next
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair, count = unique[pair_index], counts[pair_index]

            idx = i + 256
            ids = self.merge(ids, pair, idx)

            pair = tuple(pair.tolist())

            # save the merge
            #merges[pair] = idx
            merges.append((pair, idx))
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.special_tokens[vocab_size] = 'SOS'
        self.special_tokens[vocab_size+1] = 'EOS'
        self.special_tokens[vocab_size+2] = 'PAD'
        for idx, special in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        # save class variables
        self.merges = merges # used in tokenizer()
        self.vocab = vocab   # used in detokenizer()

    def tokenize(self, text):
        """
           Use merges array to map the characters in the given text
           to token integer ids. 
        """
        l = self.build_indexed_list(text)
        for pair, new_id in self.merges:
            if pair in l.index:
                self.cpu_merge(pair, new_id, l, None)
        return [node.val for node in l]

    def detokenize(self, ids):
        # Tokenizer can decode a list of integers into a string
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save(self, file: str):
        """
           Saves the merges into a file. Vocab does not need to be saved
           since it can be recovered from merges.
        """
        with open(file, 'wb') as f:
            pickle.dump((self.special_tokens,self.merges), f)

    def build_vocab(self) -> dict[str,int]:
        # Vocab is derived from merges.
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges:
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for idx,special in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")

    def load(self, file: str):
        #Load the merges from the file and then build the vocab
        with open(file, 'rb') as f:
            self.special_tokens, self.merges = pickle.load(f)
        self.build_vocab()

class TestGPU_BPE(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 260
        self.text = "aaabdaaabac"
        self.bpe = GPU_BPE()

    def test_train(self):
        """
            Given the text "aaabdaaabac", it's byte pair encoding
            are the following array of token ids [97 97 97 98 100 97 97 97 98 97 99]
            We perform four merges where at each iteration a new
            token id is used to represent the merged pair
                First merge: The pair (97 , 97) is the most frequent.
                             Let the new token id 256 represent the merged pair.
                             Thus the token ids become [256 97 98 100 256 97 98 97 99]
                Second merge: The pair (97,98) is now the most frequent.
                              Let the new token id 257 represent the merged pair.
                              Thus the token ids become [256 257 100 256 257 97 99]
                Third merge:  The pair (256 257) is now the most frequent.
                              Let the new token id 258 represent the merged pair.
                              Thus the token ids become [258 100 258 97 99]
                Fourth merge: All pairs of the same count thus, arbitarly,
                              merge the first pair (97, 99). 
                              Let the new token id 259 represent the merged pair.
                              Thus the token ids become [259 258 97 99]


        """

        self.bpe.train(self.text, self.vocab_size)
        expected_merges = [((ord('a'), ord('a')), 256), 
                            ((ord('a'), ord('b')), 257),
                            ((256, 257), 258),
                            ((ord('a'), ord('c')), 259)]
        expected_vocab = {idx: bytes([idx]) for idx in range(256)}
        expected_vocab[256] = b'aa'
        expected_vocab[257] = b'ab'
        expected_vocab[258] = b'aaab'
        expected_vocab[259] = b'ac'
        expected_vocab[260] = b'SOS'
        expected_vocab[261] = b'EOS'
        expected_vocab[262] = b'PAD'
        self.assertEqual(self.bpe.merges, expected_merges)
        self.assertEqual(self.bpe.vocab, expected_vocab)

    def test_gpu_merge(self):
        ids = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
        pair = torch.tensor([1, 2], dtype=torch.int32)
        idx = 4
        merged = self.bpe.merge(ids, pair, idx)
        self.assertTrue(torch.equal(merged, torch.tensor([4, 3, 4], dtype=torch.int32)))

    def test_tokenize_then_detokenize(self):
        # The detokenization of tokenization should
        # return the original text
        self.bpe.train(self.text, self.vocab_size)
        tokens = self.bpe.tokenize(self.text)
        detok = self.bpe.detokenize(tokens)
        self.assertEqual(detok, self.text)

    def test_tokenize_then_detokenize1(self):
        # The detokenization of tokenization should
        # return the original text
        text = "abababadasdsadaddascxzcascsaczxcasczzdsadasZxsacdcdcs"
        self.bpe.train(text, self.vocab_size)
        tokens = self.bpe.tokenize(self.text)
        detok = self.bpe.detokenize(tokens)
        self.assertEqual(detok, self.text)

    def test_save_and_load(self):
        self.bpe.train(self.text, self.vocab_size)
        self.bpe.save("test_bpe_model")
        self.assertTrue(os.path.exists("test_bpe_model"))
        bpe2 = GPU_BPE()
        bpe2.load("test_bpe_model")
        self.assertEqual(self.bpe.merges, bpe2.merges)
        self.assertEqual(self.bpe.vocab, bpe2.vocab)
        os.remove("test_bpe_model")

    def test_build_vocab(self):
        # Test the build_vocab function
        # Assumption: Vocab is cleared pior to values
        # being populated, otherwise the result of this test is invalid
        self.bpe.train(self.text, self.vocab_size)
        bpe2 = GPU_BPE()
        bpe2.merges = self.bpe.merges
        bpe2.special_tokens = self.bpe.special_tokens
        bpe2.build_vocab()
        self.assertEqual(self.bpe.vocab, bpe2.vocab)

if __name__ == "__main__":
    unittest.main()