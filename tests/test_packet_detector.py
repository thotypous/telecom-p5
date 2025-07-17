import unittest
import numpy as np
import os

from ieee80211ag.common import get_short_training_sequence
from ieee80211ag.channel import default_defect_model
from ieee80211ag.rx import packet_detector

class PacketDectorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        np.random.seed(42)  # for defect_model

    def test_packet_detector_synthetic_data(self):
        sts = get_short_training_sequence(.5)
        for i in range(128):
            pad1 = self.rng.integers(10, 200)
            pad2 = self.rng.integers(400, 800)
            signal_40mhz, _ = default_defect_model(np.concatenate((np.zeros(pad1), sts, np.zeros(pad2))))

            _, _, falling_edge_position, _ = packet_detector(signal_40mhz[::2])

            relative_falling_edge_position = falling_edge_position - pad1/2
            self.assertAlmostEqual(relative_falling_edge_position, 219, delta=9, msg=f'at iteration {i}')

    def test_packet_detector_experimental_data(self):
        data = np.load(os.path.join('tests', 'data', 'signal.npz'))
        expected_values = [-1, 222, 222, 222, 222, 222, 222, 222, 221, 222, 222, 222, 222, 221, 222, 222, 222, 222, 222, 222, 222, 222, 221, 222, 222, 222, 222, 222, 221, 221, 222, 222, 222, 222, 223, 221, 222, 222, 222, 222, 222, 221, 222, 222, 222, 221, 222, 221, 223, 222, 222, 222, 223, 222, 222, 223, 222, 222, 222, 222, 222, 223, 223, 222, 222, 222, 222, 222, 223, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 222, 221, 222, 222, 222, 222, 222, 222, 222, 221, 222, 222, 221, 222, 221, 222]
        for i, (arr, expected) in enumerate(zip(data.values(), expected_values)):
            _, _, falling_edge_position, _ = packet_detector(arr)
            self.assertEqual(falling_edge_position, expected, msg=f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
