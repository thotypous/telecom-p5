import unittest
import numpy as np

from ieee80211ag.common import scramble
from ieee80211ag.rx import descramble

class DescrambleTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_descramble(self):
        for i in range(256):
            L = self.rng.integers(7, 1024)
            initial_state = self.rng.integers(1, 128)
            arr = self.rng.integers(0, 256, L, dtype=np.uint8)
            arr[:7] = 0
            np.testing.assert_array_equal(descramble(scramble(arr, initial_state=initial_state)), arr, f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
