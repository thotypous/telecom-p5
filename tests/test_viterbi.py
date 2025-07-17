import unittest
import numpy as np

from ieee80211ag.tx import convolutional_encoder
from ieee80211ag.rx import convolutional_decoder

class ViterbiTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _add_noise(self, arr):
        arr = 2*arr.astype(float)-1
        arr += self.rng.normal(0, .4, len(arr))
        return arr

    def test_viterbi(self):
        for i in range(256):
            L = self.rng.integers(100, 5000)
            arr = self.rng.integers(0, 2, L, dtype=np.uint8)
            arr[-6:] = 0
            np.testing.assert_array_equal(convolutional_decoder(self._add_noise(convolutional_encoder(arr))), arr, f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
