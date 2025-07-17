import unittest
import numpy as np

from ieee80211ag.tx import mapper_ofdm
from ieee80211ag.rx import demapper_ofdm

class DemapperTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_demapper(self):
        for n_bpsc in 1, 2:  # testa apenas BPSK (n_bpsc=1) e QPSK (n_bpsc=2)
            for i in range(128):
                L = self.rng.integers(1, 1024)
                L -= L % n_bpsc  # assegura que o tamanho seja múltiplo de n_bpsc
                arr = self.rng.integers(0, 2, L, dtype=np.uint8)
                np.testing.assert_array_equal((demapper_ofdm(mapper_ofdm(arr, n_bpsc), n_bpsc) > 0).astype(np.uint8), arr, f'at iteration {i}, n_bpsc={n_bpsc}')

if __name__ == '__main__':
    unittest.main()
