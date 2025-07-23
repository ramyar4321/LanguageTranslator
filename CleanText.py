import unittest
import pandas as pd
import os
import re

class CleanText:
    def __init__(self, text, num_rows=None):
        """
            Read and clean file. Set number of rows are read.
            All characters are set to lower case
            and non-ASCII characters are removed.
        """
        self.text_file = text
        self.num_rows = num_rows

    def read_and_clean(self):
        eng = []
        fr = []

        # read only specific number of rows
        #if self.num_rows != None
        df = pd.read_csv(self.text_file, nrows=self.num_rows)
        df = df.reset_index()

        for index,row in df.iterrows():
          eng.append(row[1].lower())
          fr.append(row[2].lower())

        # remove non ASCII characters
        for i in range(len(eng)):
            eng[i] = re.sub(r'[^\x00-\xff]+', '', eng[i])
            fr[i] = re.sub(r'[^\x00-\xff]+', '', fr[i])

        print(f"Length of english: {len(eng)}")
        print(f"Length of french: {len(fr)}")
        return eng, fr

class TestCleanText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary CSV file for testing
        cls.test_file = "test_cleantext.csv"
        data = {
            "eng": ["Hello!", "Cafe", "Goodbye😊", "Goodbye"],
            "fr": ["Bonjour!", "Café", "Au revoir😊", "Au revoir"]
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.test_file, index=False)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_file)

    def test_read_and_clean_basic(self):
        cleaner = CleanText(self.test_file, 3)
        eng, fr = cleaner.read_and_clean()
        # Should read only 2 rows
        self.assertEqual(len(eng), 3)
        self.assertEqual(len(fr), 3)
        # Should lowercase and remove non-ASCII
        self.assertEqual(eng[0], "hello!")
        self.assertEqual(fr[0], "bonjour!")
        self.assertEqual(eng[1], "cafe")
        self.assertEqual(fr[1], "café")
        self.assertEqual(eng[2], "goodbye")
        self.assertEqual(fr[2], "au revoir")


if __name__ == "__main__":
    unittest.main()