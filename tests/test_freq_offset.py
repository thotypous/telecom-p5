import unittest
import numpy as np
import os

from ieee80211ag.common import get_short_training_sequence, get_long_training_sequence
from ieee80211ag.channel import default_defect_model, DEFECT_SETTINGS, DEFECT_MODE
from ieee80211ag.rx import detect_frequency_offsets

class FreqOffsetTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        np.random.seed(42)  # for defect_model

    def test_freq_offset_synthetic_data(self):
        assert DEFECT_MODE['Freq_Offset'] == 1
        model_freq_offset = DEFECT_SETTINGS['FrequencyOffset']

        sts = get_short_training_sequence(.5)
        lts, _ = get_long_training_sequence(.5)

        for i in range(128):
            pad1 = self.rng.integers(10, 200)
            pad2 = self.rng.integers(400, 800)
            signal_40mhz, _ = default_defect_model(np.concatenate((np.zeros(pad1), sts, lts, np.zeros(pad2))))
            signal_20mhz = signal_40mhz[::2]

            falling_edge_position = pad1//2 + 219
            freq_offset = detect_frequency_offsets(signal_20mhz, falling_edge_position)
            coarse_offset = freq_offset[0]
            self.assertAlmostEqual(coarse_offset, model_freq_offset, delta=60e3, msg=f'at iteration {i}')

            n = np.arange(len(signal_20mhz))
            signal_20mhz *= np.exp(-1j * 2 * np.pi * n * coarse_offset / 20e6)
            freq_offset = detect_frequency_offsets(signal_20mhz, falling_edge_position)
            fine_offset = freq_offset[1]
            self.assertAlmostEqual(coarse_offset + fine_offset, model_freq_offset, delta=10e3, msg=f'at iteration {i}')

    def test_freq_offset_experimental_data(self):
        data = np.load(os.path.join('tests', 'data', 'signal.npz'))

        mean_measured_falling_edge_position = 222
        mean_measured_offset = -17858

        for i, arr in enumerate(data.values()):
            if i == 0:
                continue  # pula o primeiro registro (propositalmente corrompido)

            freq_offset = detect_frequency_offsets(arr, mean_measured_falling_edge_position)
            coarse_offset = freq_offset[0]

            self.assertAlmostEqual(coarse_offset, mean_measured_offset, delta=8e3, msg=f'at iteration {i}')

            n = np.arange(len(arr))
            arr *= np.exp(-1j * 2 * np.pi * n * coarse_offset / 20e6)
            freq_offset = detect_frequency_offsets(arr, mean_measured_falling_edge_position)
            fine_offset = freq_offset[1]

            self.assertAlmostEqual(coarse_offset + fine_offset, mean_measured_offset, delta=2.5e3, msg=f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
