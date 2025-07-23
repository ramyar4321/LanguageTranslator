import unittest

class LL_Node:
    """
        A node in a doubly linked list.
        The node's value is one or more characters.
    """
    __slots__ = 'val', 'prev', 'next'
    def __init__(self, val, prev, next):
        self.val, self.prev, self.next = val, prev, next

    def delete(self):
        if self.prev is not None:
            self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev
        self.next = self.prev = None

class IndexedList:

    """
        This doubly linked list stores the characters (or bytes)
        of the input text, and its index dictionary
        maps pairs of tokens to the nodes where they appear.
    """

    def __init__(self, text: str):
        """
            The index stores a list of nodes for each pair.
            As merges occur, some of these nodes might become "stale"
            (no longer represent the original pair).
            The downstream merge must check for staleness.
        """
        self.index = {}
        text = iter(text)
        x = next(text)
        self.start = prev_node = LL_Node(x, None, None)
        for y in text:
            prev_node.next = node = LL_Node(y, prev_node, None)
            self.add_to_index((x, y), prev_node)
            x, prev_node = y, node

    def __iter__(self):
        # Interator for the doubly linked list
        node = self.start
        while node is not None:
            yield node
            node = node.next

    def update_index(self, node: LL_Node) -> None:
        # Update index before/after node.
        if node.prev is not None:
            self.add_to_index((node.prev.val, node.val), node.prev)
        if node.next is not None:
            self.add_to_index((node.val, node.next.val), node)

    def add_to_index(self, pair: str, node: LL_Node) ->  None:
        # Add node to the list of nodes for given pair.
        self.index.setdefault(pair, []).append(node)

class TestIndexedList(unittest.TestCase):

    def test_init_and_structure(self):
        s = "aabcaade"
        ilist = IndexedList(s)
        # Check linked list values
        node = ilist.start
        vals = []
        while node:
            vals.append(node.val)
            node = node.next
        self.assertEqual(vals, list(s))
        # Check index for all pairs
        for i in range(len(s) - 1):
            pair = (s[i], s[i+1])
            self.assertIn(pair, ilist.index)
            # The node for the pair should have val == s[i]
            self.assertTrue(all(n.val == s[i] for n in ilist.index[pair]))

    def test_init_single_char(self):
        s = "x"
        ilist = IndexedList(s)
        self.assertEqual(ilist.start.val, "x")
        self.assertIsNone(ilist.start.prev)
        self.assertIsNone(ilist.start.next)
        self.assertEqual(ilist.index, {})

    def test_iter(self):
        s = "wxyz"
        ilist = IndexedList(s)
        vals = [node.val for node in ilist]
        self.assertEqual(vals, list(s))

    def test_add_to_index(self):
        ilist = IndexedList("ab")
        node = LL_Node("x", None, None)
        pair = ("a", "b")
        ilist.add_to_index(pair, node)
        self.assertIn(pair, ilist.index)
        self.assertIn(node, ilist.index[pair])


    def test_update_index1(self):
        ilist = IndexedList("abcd")
        # Merge 'b' and 'c' into one node
        node_b = ilist.start.next
        node_c = node_b.next
        node_b.val += node_c.val  # 'bc'
        node_b.next = node_c.next
        # Remove node_z from the list
        node_c.prev = None
        # Update index for node_b
        ilist.update_index(node_b)
        # Should update index for ('a','bz')
        print(ilist.index)
        self.assertIn(('a', 'bc'), ilist.index)
        self.assertIn(node_b, ilist.index[('bc', 'd')])

if __name__ == "__main__":
    unittest.main()


