import unittest
import numpy as np
import os

from ieee80211ag.rx import ofdm_receiver, decode_signal_field, decode_data_symbols

class IntegrationTest(unittest.TestCase):
    def test_integration_experimental_data(self):
        signal_data = np.load(os.path.join('tests', 'data', 'signal.npz'))
        frames_data = np.load(os.path.join('tests', 'data', 'frames.npz'))
        for i, (arr, expected) in enumerate(zip(signal_data.values(), frames_data.values())):
            corrected_symbols = ofdm_receiver(arr,
                sample_advance=0,
                correct_frequency_offset=1,
                number_of_ofdm_symbols=1000,
                use_max_ratio_combining=1)

            if corrected_symbols.size == 0:
                received_mac_frame_bytes = np.array([], dtype=np.uint8)
            else:
                decoded_params = decode_signal_field(corrected_symbols[:48])
                self.assertEqual(decoded_params['parity_ok'], True, msg=f'at iteration {i}')
                self.assertEqual(decoded_params['tail_ok'], True, msg=f'at iteration {i}')

                # todos os quadros da gravação são de 12Mbps
                self.assertEqual(decoded_params['rate_info']['Mbps'], 12, msg=f'at iteration {i}')

                received_mac_frame_bytes, tail_ok, crc_ok = decode_data_symbols(corrected_symbols[48:], decoded_params['rate_info'], decoded_params['length'])
                self.assertEqual(tail_ok, True, msg=f'at iteration {i}')
                self.assertEqual(crc_ok, True, msg=f'at iteration {i}')

            np.testing.assert_array_equal(received_mac_frame_bytes, expected, f'at iteration {i}')

if __name__ == '__main__':
    unittest.main()
