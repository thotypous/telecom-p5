import unittest
import numpy as np
import os

from ieee80211ag.common import get_short_training_sequence, get_long_training_sequence
from ieee80211ag.channel import default_defect_model, DEFECT_SETTINGS, DEFECT_MODE
from ieee80211ag.rx import long_symbol_correlator

class LongSymbolCorrTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        np.random.seed(42)  # for defect_model

    def test_long_symbol_corr_synthetic_data(self):
        model_freq_offset = DEFECT_SETTINGS['FrequencyOffset']

        sts = get_short_training_sequence(.5)
        lts, _ = get_long_training_sequence(.5)
        lts_20mhz, _ = get_long_training_sequence(1)

        for i in range(128):
            pad1 = self.rng.integers(10, 200)
            pad2 = self.rng.integers(400, 800)
            signal_40mhz, _ = default_defect_model(np.concatenate((np.zeros(pad1), sts, lts, np.zeros(pad2))))
            signal_20mhz = signal_40mhz[::2]

            falling_edge_position = pad1//2 + 219
            n = np.arange(len(signal_20mhz))
            signal_20mhz *= np.exp(-1j * 2 * np.pi * n * model_freq_offset / 20e6)

            long_training_symbol = lts_20mhz[32:96]
            _, lt_peak_position, _ = long_symbol_correlator(long_training_symbol, signal_20mhz, falling_edge_position)
            
            self.assertAlmostEqual(lt_peak_position - pad1/2, 322, delta=6, msg=f'at iteration {i}')

    def test_long_symbol_corr_experimental_data(self):
        data = np.load(os.path.join('tests', 'data', 'signal.npz'))

        mean_measured_falling_edge_position = 222
        mean_measured_offset = -17858

        lts_20mhz, _ = get_long_training_sequence(1)

        for i, arr in enumerate(data.values()):
            if i == 0:
                continue  # pula o primeiro registro (propositalmente corrompido)

            n = np.arange(len(arr))
            arr *= np.exp(-1j * 2 * np.pi * n * mean_measured_offset / 20e6)

            long_training_symbol = lts_20mhz[32:96]
            _, lt_peak_position, _ = long_symbol_correlator(long_training_symbol, arr, mean_measured_falling_edge_position)

            self.assertAlmostEqual(lt_peak_position, 305.4, delta=.7, msg=f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
