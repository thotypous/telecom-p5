import unittest
import numpy as np

from ieee80211ag.tx import interleave
from ieee80211ag.rx import create_deinterleaving_pattern
from ieee80211ag.common import RATE_MAP

class DeinterleaverTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_deinterleaver(self):
        for info in RATE_MAP.values():
            n_cbps = info['n_cbps']
            n_bpsc = info['n_bpsc']
            pattern = create_deinterleaving_pattern(n_cbps, n_bpsc)
            for i in range(16):
                arr = self.rng.integers(0, 2, n_cbps, dtype=np.uint8)
                np.testing.assert_array_equal(interleave(arr, n_cbps, n_bpsc)[pattern], arr, f'at iteration {i}, n_cbps={n_cbps}, n_bpsc={n_bpsc}')

if __name__ == '__main__':
    unittest.main()
