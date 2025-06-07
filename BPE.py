# Code adapted from
# https://yanivle.github.io/ai/2024/02/23/fast_minbpe.html
from CountPair import CountPair
from IndexedList import IndexedList
from typing import Dict
import unittest
import os
import pickle

class BPE:
    """
       Byte-Pair Econding tokenizer
    """

    def __init__(self, vocab_size: int):
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab_size = vocab_size+256
        self.special_tokens = {'SOS': self.vocab_size,
                                'PAD': (self.vocab_size+1),
                                'EOS': self.vocab_size+2}

    def build_indexed_list(self, text: str) -> IndexedList:  
        # Create an IndexedList with the encoded bytes.
        return IndexedList(t for t in text.encode('utf-8'))


    def init_pairs_stats(self, text: str) -> CountPair:  
        """
           Initialize a CountPair with all overlapping pairs.
           For text "aaabd" the CountPair will contain: {(a,a): 2, (a, b): 1, (b, d): 1}
        """
        return CountPair(t for t in text.encode('utf-8'))


    def merge(self, pair: tuple[str, str], new_id: int, 
                indexed_list: IndexedList, stats: CountPair=None) -> None:
        """ 
            Find all instances of the given pair in the index list 
            and merge them to form the new_id. If this function
            is called by the train function, then the CountPair is
            updated for each merge.

            This function performs M merges in the IndexedList
            where each merge is O(1). For each merge, CountPair
            is updated which is also O(log L) where L is the training text
            length, thus this function has a time complexity of O(M)
            
        """
        # O(M)
        for node in indexed_list.index[pair]:
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


    def train(self, text: str) -> tuple[list[tuple[str,int]], Dict[str, int]]:
        """
           Form the merges from the text.
           Iteratively find the most common pair of token ids and merge them
           by replacing them with a new token id and then update the
           count of pairs. Add the pair and its corresponding new token id
           to the merges array.

           The algorithm and complexity are as follows:

           1. Compute_pair_counts # O(L)
           2. for i=0 to n
           3.    find_most_common_pair #O(1)
           4.    merge_most_common_pair_and_update_pair_counts O(M_i)

           where L is the length of the text, n is 
           the vocab_size for, M_i is the number of
           merges and count updates for the ith pair merge for 
           M_0 + M_1 + ... M_(n-1) <= L. For n << L, the time complexity
           is O(L)

           Arg:
              text: str Used to create the merge vocab 
        """
        self.merges.clear()
        self.vocab.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        print(f'Training tokenizer on text of length {len(text):,} with vocab of size {self.vocab_size:,}.')
        n_merges = self.vocab_size - 256
        indexed_list = self.build_indexed_list(text)
        stats = self.init_pairs_stats(text)
        for i in range(n_merges):
            if not stats: break  # Stop if we don't have any pairs (we should probably stop earlier).
            top_pair = stats.most_common
            new_id = len(self.vocab)
            self.merges.append((top_pair, new_id))
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            self.merge(top_pair, new_id, indexed_list, stats)


    def tokenize(self, text: str, ) -> list[int]:
        """
           Use merges array to map the characters in the given text
           to token integer ids. 
        """
        l = self.build_indexed_list(text)
        for pair, new_id in self.merges:
            if pair in l.index:
                self.merge(pair, new_id, l, None)
        return [node.val for node in l]


    def detokenize(self, seq: list[int]) -> list[str]:
        """
           Use vocabulary hashtable to map token integer ids
           to their corresponding merged character(s).

           Arg: 
              seq: str Sequence of token ids
              vocab: Dict Hashtable mapping token ids integer keys
                     to merged character values
        """
        return b''.join((self.vocab[t] for t in seq)).decode('utf-8')

    def save(self, file: str):
        """
           Saves the merges into a file. Vocab does not need to be saved
           since it can be recovered from merges.
        """
        with open(file, 'wb') as f:
            pickle.dump(self.merges, f)

    def build_vocab(self) -> Dict[str,int]:
        # Vocab is derived from merges.
        self.vocab.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges:
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def load(self, file: str):
        #Load the merges from the file and then build the vocab
        with open(file, 'rb') as f:
            self.merges = pickle.load(f)
        self.build_vocab()

class TestBPE(unittest.TestCase):

    def setUp(self):
        #self.vocab_size = 260
        self.vocab_size = 4
        self.bpe = BPE(self.vocab_size)
        self.text = "aaabdaaabac"

    def test_merge_with_no_stats(self):
        ilist = self.bpe.build_indexed_list(self.text)
        pair = (ord('a'), ord('a'))
        new_id = 256
        # Should not raise even if stats is None
        self.bpe.merge(pair, new_id, ilist, stats=None)
        found = any(node.val == new_id for node in ilist)
        self.assertTrue(found)

    def test_merge_with_stats(self):
        # Test that merge actually merges pairs in the indexed list
        ilist = self.bpe.build_indexed_list(self.text)
        stats = self.bpe.init_pairs_stats(self.text)
        pair = stats.most_common
        new_id = 256
        self.bpe.merge(pair, new_id, ilist, stats)
        # After merging, there should be a node with value new_id
        found = any(node.val == new_id for node in ilist)
        self.assertTrue(found)

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
                              merge the first pair (258, 100). 
                              Let the new token id 259 represent the merged pair.
                              Thus the token ids become [259 258 97 99]


        """

        self.bpe.train(self.text)
        expected_merges = [((ord('a'), ord('a')), 256), 
                            ((ord('a'), ord('b')), 257),
                            ((256, 257), 258),
                            ((258, ord('d')), 259)]
        expected_vocab = {idx: bytes([idx]) for idx in range(256)}
        expected_vocab[256] = b'aa'
        expected_vocab[257] = b'ab'
        expected_vocab[258] = b'aaab'
        expected_vocab[259] = b'aaabd'
        self.assertEqual(self.bpe.merges, expected_merges)
        self.assertEqual(self.bpe.vocab, expected_vocab)

    def test_tokenize_then_detokenize(self):
        # The detokenization of tokenization should
        # return the original text
        self.bpe.train(self.text)
        tokens = self.bpe.tokenize(self.text)
        detok = self.bpe.detokenize(tokens)
        self.assertEqual(detok, self.text)

    def test_save_and_load(self):
        self.bpe.train(self.text)
        self.bpe.save("test_bpe_model")
        self.assertTrue(os.path.exists("test_bpe_model"))
        bpe2 = BPE(self.vocab_size)
        bpe2.load("test_bpe_model")
        self.assertEqual(self.bpe.merges, bpe2.merges)
        self.assertEqual(self.bpe.vocab, bpe2.vocab)
        os.remove("test_bpe_model")

    def test_build_vocab(self):
        # Test the build_vocab function
        # Assumption: Vocab is cleared pior to values
        # being populated, otherwise the result of this test is invalid
        self.bpe.train(self.text)
        bpe2 = BPE(self.vocab_size)
        bpe2.train(self.text)
        bpe2.build_vocab()
        self.assertEqual(self.bpe.vocab, bpe2.vocab)

if __name__ == "__main__":
    unittest.main()


