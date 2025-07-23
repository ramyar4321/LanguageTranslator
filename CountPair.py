from itertools import pairwise
import unittest

class MH_Node:
    __slots__ = 'count', 'val', 'pos'

    def __init__(self, count: int, val: str, pos: int):
        self.count = count
        self.val = val
        self.pos = pos

    @property
    def key(self):  # key for comparisons
        return self.count
        # Breaking ties explicitly, forcing more heap update, results in a significant slowdown:
        # return (self.count, self.val, self.pos)

    def __lt__(self, other):
        return self.key < other.key


class CountPair:
    """
       This class is a modified max heap data structure used to
       keep count of pairs of character(s) in a given text.
       The max heap is implemented as an array and
       a hashtable is used to map the pair to the nodes in the heap.
    """
    def __init__(self, text: str):
        self.l = []  # A heap of nodes.
        self.d = {}  # A map from value to its node.

        # Construct the max heap from the text
        for a,b in pairwise(text):
            self.add((a,b), 1)


    def add(self, pair: tuple[str, str], count=1):
        node = self.d.get(pair)
        if node is None:
            node = self.d[pair] = MH_Node(0, pair, len(self.l))
            self.l.append(node)
        node.count += count
        self.pair_increased(node.pos)

    def remove(self, pair: tuple[str: str], count=1):
        node = self.d[pair]
        node.count -= count
        self.pair_decreased(node.pos)
        # We could actually remove pairs with 0-count from the list, but
        # since for some scores its helpful to have pairs with arbitrary
        # counts, including negative, we're never actually removing pairs.

    def count(self, pair: tuple[str, str]):
        # Return the count of the pair
        if pair not in self.d: return 0
        return self.d[pair].count

    @property
    def most_common(self):
        # Return the pair with the highest count
        return self.l[0].val

    def __bool__(self):
        return bool(self.l)

    # The below functions maintain the heap property when a given pair count
    # is increased or decreased:

    def pair_increased(self, pos: int):
        # Adapted from heapq._siftdown_max.
        node = self.l[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self.l[parentpos]
            if parent < node:
                self.l[pos] = parent
                parent.pos = pos
                pos = parentpos
                continue
            break
        self.l[pos] = node
        node.pos = pos

    def pair_decreased(self, pos: int):
        # Adapted from heapq._siftup_max.
        endpos = len(self.l)
        node = self.l[pos]
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of larger child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.l[rightpos] < self.l[childpos]:
                childpos = rightpos
            childnode = self.l[childpos]
            if node < childnode:  # Move the larger child up.
                self.l[pos] = childnode
                childnode.pos = pos
                pos = childpos
                childpos = 2 * pos + 1
            else:
                break
        self.l[pos] = node
        node.pos = pos


class TestCountPair(unittest.TestCase):

    def test_init_and_add(self):
        cp = CountPair("aabbc")
        # Should have pairs: ('a','a'), ('a','b'), ('b','b'), ('b','c')
        self.assertEqual(cp.count(('a','a')), 1)
        self.assertEqual(cp.count(('a','b')), 1)
        self.assertEqual(cp.count(('b','b')), 1)
        self.assertEqual(cp.count(('b','c')), 1)
        self.assertEqual(cp.count(('c','a')), 0)  # Not present

    def test_add_increases_count(self):
        cp = CountPair("ab")
        cp.add(('a','b'), 1)
        self.assertEqual(cp.count(('a','b')), 2)

    def test_remove_decreases_count(self):
        cp = CountPair("aab")
        cp.remove(('a','a'), 1)
        self.assertEqual(cp.count(('a','a')), 0)
        cp.remove(('a','b'), 1)
        self.assertEqual(cp.count(('a','b')), 0)

    def test_most_common(self):
        cp = CountPair("aabbb")
        # ('b','b') appears twice, others once
        self.assertEqual(cp.most_common, ('b','b'))
        cp.add(('a','a'), 3)
        self.assertEqual(cp.most_common, ('a','a'))

    def test_bool(self):
        cp = CountPair("ab")
        self.assertTrue(cp)
        cp.remove(('a','b'), 1)
        self.assertTrue(cp)  # Still has the pair with count 0
        # Remove all pairs
        for node in cp.l:
            node.count = 0
        self.assertTrue(cp)  # Still True because list is not empty

    def test_heap_property(self):
        cp = CountPair("aabbb")
        # Increase ('a','b') to be most common
        cp.add(('a','b'), 5)
        self.assertEqual(cp.most_common, ('a','b'))
        # Decrease ('a','b') to be less than ('b','b')
        cp.remove(('a','b'), 6)
        self.assertEqual(cp.most_common, ('b','b'))

if __name__ == "__main__":
    unittest.main()

