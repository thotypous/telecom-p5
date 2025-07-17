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

    def _add_low_confidence_sign_errors(self, arr, positions):
        soft_bits = 2*arr.astype(float)-1
        soft_bits *= 3
        soft_bits[positions] = -np.sign(soft_bits[positions])*.1
        return soft_bits

    def _hard_quantize(self, soft_bits):
        return np.where(soft_bits >= 0, 1., -1.)

    def test_viterbi(self):
        for i in range(256):
            L = self.rng.integers(100, 5000)
            arr = self.rng.integers(0, 2, L, dtype=np.uint8)
            arr[-6:] = 0
            np.testing.assert_array_equal(convolutional_decoder(self._add_noise(convolutional_encoder(arr))), arr, f'at iteration {i}')

    def test_viterbi_uses_soft_reliability(self):
        cases = [
            (
                np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
                          0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
                [42, 32, 11, 43],
            ),
            (
                np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                          1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
                [32, 13, 46, 45],
            ),
            (
                np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
                          0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
                [39, 46, 2, 45],
            ),
        ]

        for arr, error_positions in cases:
            with self.subTest(error_positions=error_positions):
                soft_bits = self._add_low_confidence_sign_errors(convolutional_encoder(arr), error_positions)
                np.testing.assert_array_equal(convolutional_decoder(soft_bits), arr)

                hard_decoded = convolutional_decoder(self._hard_quantize(soft_bits))
                self.assertFalse(np.array_equal(hard_decoded, arr))

    def test_encoder_matches_course_vector(self):
        arr = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        expected = np.array([
            1, 1, 0, 1, 0, 0, 0, 1,
            1, 0, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 1, 1, 0, 1, 1,
        ])
        np.testing.assert_array_equal(convolutional_encoder(arr), expected)

if __name__ == '__main__':
    unittest.main()
