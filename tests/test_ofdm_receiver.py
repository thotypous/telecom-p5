import unittest
from unittest.mock import patch
import numpy as np
import os

from ieee80211ag.channel import default_defect_model, DEFECT_SETTINGS
from ieee80211ag.tx import ofdm_transmitter
from ieee80211ag.rx import ofdm_receiver

def mocked_packet_detector(rx_input):
    return None, None, 177, None

def mocked_detect_frequency_offsets(rx_input, falling_edge_position):
    return 0, DEFECT_SETTINGS['FrequencyOffset']

def mocked_long_symbol_correlator(long_training_symbol, rx_waveform, falling_edge_position):
    return None, 262, None

@patch('ieee80211ag.rx.packet_detector', mocked_packet_detector)
@patch('ieee80211ag.rx.detect_frequency_offsets', mocked_detect_frequency_offsets)
@patch('ieee80211ag.rx.long_symbol_correlator', mocked_long_symbol_correlator)
class OFDMReceiverTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        np.random.seed(42)  # for defect_model

    def test_ofdm_receiver_synthetic_data(self):
        results = []
        for i in range(64):
            L = self.rng.integers(100, 1000)

            mac_frame_bytes = np.random.randint(0, 256, L, dtype=np.uint8)
            sample_output, tx_symbol_stream = ofdm_transmitter(mac_frame_bytes, rate_key=0b0101, transmitter_choice=1)
            tx_output_clean = np.concatenate((np.zeros(10), sample_output, np.zeros(20)))

            tx_output, _ = default_defect_model(tx_output_clean)

            rx_waveform_20mhz = tx_output[::2]
            corrected_symbols = ofdm_receiver(rx_waveform_20mhz,
                sample_advance=1,
                correct_frequency_offset=1,
                number_of_ofdm_symbols=1000,
                use_max_ratio_combining=1)

            error_vectors = tx_symbol_stream - corrected_symbols
            average_error_vector_power = np.mean(np.abs(error_vectors)**2)

            # Evita log de zero
            if average_error_vector_power == 0:
                average_error_vector_power = 1e-12

            evm = 10 * np.log10(average_error_vector_power / 1)
            self.assertLess(evm, -12, msg=f'at iteration {i}')
            results.append(evm)

        self.assertLess(np.mean(results) + np.std(results), -20)

if __name__ == '__main__':
    unittest.main()
