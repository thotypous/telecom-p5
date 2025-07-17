# =============================================================================
# Funções do Transmissor (TX)
# =============================================================================

import numpy as np
from viterbi import Viterbi
import binascii

from .common import *

_INTERLEAVING_PATTERNS = {} # Cache para os padrões

def ofdm_transmitter(mac_frame_bytes, rate_key=0b0101, transmitter_choice=1):
    """
    Gera uma forma de onda de pacote 802.11a/g completa, incluindo os campos
    SIGNAL e DATA, codificados e modulados conforme a norma.
    Referência: Livro-texto, Seção 7.2, Figura 7-17 (visão geral do transmissor).
    """
    # 1. Incluir CRC ao final de mac_frame_bytes para obter psdu_bytes
    crc = (binascii.crc32(mac_frame_bytes) & 0xFFFFFFFF).to_bytes(4, 'little')
    psdu_bytes = np.concatenate((mac_frame_bytes, np.frombuffer(crc, dtype=np.uint8))).tobytes()

    # 2. Obter preâmbulo (sequências de treinamento curta e longa).
    # A escolha do passo (step) determina a taxa de amostragem (20MHz ou 40MHz).
    step = 0.5 if transmitter_choice else 1
    sample_rate = 20e6 if transmitter_choice else 40e6
    short_training_sequence = get_short_training_sequence(step)
    long_training_sequence, _ = get_long_training_sequence(step)

    # 3. Gerar o Campo SIGNAL.
    # Este campo é sempre BPSK 1/2 e contém a taxa e o comprimento do resto do pacote.
    # Referência: Norma IEEE 802.11a, Seção 17.3.4.
    signal_symbols = create_signal_field(len(psdu_bytes), rate_key)
    signal_waveform = (ifft128_gi if transmitter_choice else ifft_gi)(signal_symbols, start_symbol_index=0) # p_0 é usado

    # 4. Gerar Símbolos de DADOS.
    data_symbols = encode_data_field(psdu_bytes, rate_key)
    data_waveform = (ifft128_gi if transmitter_choice else ifft_gi)(data_symbols, start_symbol_index=1) # p_1 em diante

    # 5. Montar pacote completo: Preâmbulo + SIGNAL + DATA
    packet = np.concatenate((short_training_sequence, long_training_sequence, signal_waveform, data_waveform))

    # 6. Upsampling por software se a opção não-IFFT128 for escolhida.
    if not transmitter_choice:
        packet_zero_stuffed = np.zeros(2 * len(packet), dtype=complex)
        packet_zero_stuffed[::2] = packet

        # Calcular coeficientes do filtro de meia banda.
        # Um filtro de meia banda é eficiente para interpolação por um fator de 2.
        # Referência: Livro-texto, Seção 3.4.3 e Figura 7-35.
        N = 31
        n = np.arange(N)
        arg = n / 2 - (N - 1) / 4
        h = np.sinc(arg) * (hann(N + 2, sym=False)[1:N+1]**0.5)

        # Operação de filtragem de meia banda para completar o upsampling.
        packet = lfilter(h, [1.0], np.concatenate((packet_zero_stuffed, np.zeros(100))))

    return packet, np.concatenate((signal_symbols, data_symbols))

def create_signal_field(psdu_length_bytes, rate_key=0b0101):
    """
    Cria os 48 símbolos BPSK para o campo SIGNAL do preâmbulo.
    Referência: Norma IEEE 802.11a, Seção 17.3.4 e Figura 111.
    Referência gr-ieee802-11: `lib/signal_field_impl.cc`
    """
    # Bits 0-3: RATE
    rate_bits = np.array([int(b) for b in format(rate_key, '04b')])

    # Bit 4: Reservado
    reserved_bit = np.array([0])

    # Bits 5-16: LENGTH (12 bits, LSB primeiro)
    length_bits = np.array([int(b) for b in format(psdu_length_bytes, '012b')[::-1]])

    # Monta os primeiros 17 bits
    info_bits = np.concatenate((rate_bits, reserved_bit, length_bits))

    # Bit 17: Paridade par para os primeiros 17 bits (0-16)
    parity = np.sum(info_bits) % 2

    # Bits 18-23: Cauda de zeros
    tail_bits = np.zeros(6, dtype=int)

    signal_field_24_bits = np.concatenate((info_bits, [parity], tail_bits))

    # Codificação convolucional (sempre taxa 1/2) -> 48 bits
    encoded_bits = convolutional_encoder(signal_field_24_bits)

    # Interleaving (sempre BPSK, n_bpsc=1, n_cbps=48)
    interleaved_bits = interleave(encoded_bits, 48, 1)

    # Mapeamento BPSK
    signal_symbols = 2 * interleaved_bits - 1

    return signal_symbols

def encode_data_field(psdu_bytes, rate_key=0b0101):
    """
    Codifica o campo de dados (PSDU) em símbolos QPSK.
    Referência: Norma IEEE 802.11a, Seção 17.3.5.
    """
    rate_params = RATE_MAP[rate_key]
    n_dbps = rate_params['n_dbps']
    n_cbps = rate_params['n_cbps']
    n_bpsc = rate_params['n_bpsc']

    # Monta o campo DATA: SERVICE (16 bits) + PSDU + TAIL (6 bits)
    service_bits = np.zeros(16, dtype=int)
    psdu_bits = np.unpackbits(np.frombuffer(psdu_bytes, dtype=np.uint8), bitorder='little')
    tail_bits = np.zeros(6, dtype=int)

    total_data_bits = len(service_bits) + len(psdu_bits) + len(tail_bits)
    num_ofdm_symbols = int(np.ceil(total_data_bits / n_dbps))

    # Adiciona bits de preenchimento (padding) para completar o último símbolo OFDM.
    num_pad_bits = num_ofdm_symbols * n_dbps - total_data_bits
    pad_bits = np.zeros(num_pad_bits, dtype=int)

    data_to_encode = np.concatenate((service_bits, psdu_bits, tail_bits, pad_bits))

    # Scrambling (com estado inicial pseudoaleatório)
    initial_scrambler_state = np.random.randint(1, 128)
    scrambled_bits = scramble(data_to_encode, initial_scrambler_state)
    # Zera bits da cauda do codificador convolucional, para que a cauda seja determinística
    scrambled_bits[len(service_bits) + len(psdu_bits):len(service_bits) + len(psdu_bits) + len(tail_bits)] = tail_bits

    # Codificação convolucional
    encoded_bits = convolutional_encoder(scrambled_bits)

    # Puncturing poderia ser adicionado aqui para taxas de código > 1/2

    # Interleaving e Mapeamento por símbolo OFDM
    data_symbols = np.array([], dtype=complex)
    for i in range(num_ofdm_symbols):
        start, stop = i * n_cbps, (i + 1) * n_cbps
        symbol_encoded_bits = encoded_bits[start:stop]

        symbol_interleaved_bits = interleave(symbol_encoded_bits, n_cbps, n_bpsc)

        # Mapeia os bits entrelaçados para a constelação apropriada.
        symbol = mapper_ofdm(symbol_interleaved_bits, n_bpsc)

        data_symbols = np.concatenate((data_symbols, symbol))

    return data_symbols

def convolutional_encoder(bits):
    """
    Codifica bits usando o codificador convolucional padrão IEEE 802.11 (taxa 1/2, K=7).
    Referência: Norma IEEE 802.11a, Seção 17.3.5.5.
    Referência Livro-texto: Seção 5.6.3.
    Referência gr-ieee802-11: `lib/viterbi_decoder/base.h` e implementações.
    """
    trellis = Viterbi(7, [0o133, 0o171])
    return np.array(trellis.encode(bits), dtype=int)

def create_interleaving_pattern(n_cbps, n_bpsc):
    """
    Cria e armazena em cache o padrão de interleaving para um dado
    número de bits codificados por símbolo (N_CBPS) e por subportadora (N_BPSC).
    Referência gr-ieee802-11: `lib/utils.cc`, função `interleave`.
    """
    if (n_cbps, n_bpsc) in _INTERLEAVING_PATTERNS:
        return _INTERLEAVING_PATTERNS[(n_cbps, n_bpsc)]

    # Primeira permutação (inversa de i para k)
    k = np.arange(n_cbps)
    i = (n_cbps // 16) * (k % 16) + (k // 16)

    # Segunda permutação (inversa de j para i)
    s = max(n_bpsc // 2, 1)
    interleave_map = np.zeros(n_cbps, dtype=int)
    for i_idx in range(n_cbps):
        j = s * (i_idx // s) + (i_idx + n_cbps - (16 * i_idx // n_cbps)) % s
        interleave_map[i_idx] = j

    pattern = np.argsort(interleave_map[i])
    _INTERLEAVING_PATTERNS[(n_cbps, n_bpsc)] = pattern
    return pattern

def interleave(bits, n_cbps, n_bpsc):
    """Aplica o padrão de interleaving aos bits."""
    pattern = create_interleaving_pattern(n_cbps, n_bpsc)
    return bits[pattern]

def mapper_ofdm(input_bits, n_bpsc):
    """
    Mapeia uma sequência de bits para símbolos de constelação (BPSK, QPSK, etc.).
    A escala dos pontos da constelação é normalizada para manter a potência média igual a 1.
    Referência: Livro-texto, Seção 5.2.1 e Figura 5-13/5-14.
    Referência Norma IEEE 802.11a: Seção 17.3.5.7 e Tabela 81.
    Referência gr-ieee802-11: `lib/constellations_impl.cc`.
    """
    num_symbols = len(input_bits) // n_bpsc
    output_symbols = np.zeros(num_symbols, dtype=complex)

    # Itera sobre os grupos de bits e mapeia para os símbolos correspondentes.
    for i in range(num_symbols):
        start = i * n_bpsc
        stop = start + n_bpsc
        bit_group = input_bits[start:stop]

        if n_bpsc == 1: # BPSK
            symbol = BPSK_LUT[bit_group[0]]
        elif n_bpsc == 2: # QPSK
            symbol = QPSK_LUT[bit_group[0]] + 1j * QPSK_LUT[bit_group[1]]
        elif n_bpsc == 4: # 16-QAM
            # O mapeamento Gray é usado para minimizar erros de bit.
            idx_i = bit_group[0] * 2 + bit_group[1]
            idx_q = bit_group[2] * 2 + bit_group[3]
            symbol = QAM16_LUT[idx_i] + 1j * QAM16_LUT[idx_q]
        elif n_bpsc == 6: # 64-QAM
            idx_i = bit_group[0] * 4 + bit_group[1] * 2 + bit_group[2]
            idx_q = bit_group[3] * 4 + bit_group[4] * 2 + bit_group[5]
            symbol = QAM64_LUT[idx_i] + 1j * QAM64_LUT[idx_q]
        else:
            raise ValueError(f"n_bpsc={n_bpsc} not supported")

        output_symbols[i] = symbol

    return output_symbols

def ifft_gi(symbol_stream, start_symbol_index=0):
    """
    Executa a IFFT e adiciona o intervalo de guarda (GI) para um fluxo de símbolos.
    Mapeia 48 símbolos de dados e 4 pilotos para as subportadoras corretas de uma IFFT de 64 pontos.
    Referência: Norma IEEE 802.11a, Seção 17.3.5.9, Figura 7-30 do Livro-texto.
    Referência gr-ieee802-11: `digital_ofdm_carrier_allocator_cvc`.
    """
    num_symbols = len(symbol_stream) // 48
    payload = np.zeros(num_symbols * 80, dtype=complex)
    ifft_input = np.zeros(64, dtype=complex)

    for i in range(num_symbols):
        # A polaridade dos pilotos muda a cada símbolo para espalhar a energia.
        symbol_idx = start_symbol_index + i

        start_symbol, stop_symbol = i * 48, (i + 1) * 48
        ifft_input[:] = 0
        ifft_input[DATA_CARRIERS_IDX] = symbol_stream[start_symbol:stop_symbol]
        # Os pilotos são modulados por BPSK e multiplicados pela polaridade.
        ifft_input[PILOT_CARRIERS_IDX] = PILOT_BASE_POLARITY * PILOT_POLARITY[symbol_idx % 127]

        # A IFFT converte o sinal do domínio da frequência para o domínio do tempo.
        ifft_output = np.fft.ifft(ifft_input)

        # Adiciona o Intervalo de Guarda (GI) ou Prefixo Cíclico.
        # O GI é uma cópia das últimas 16 amostras do símbolo IFFT, prefixado ao símbolo.
        # Isso mitiga a interferência intersimbólica (ISI) causada por múltiplos percursos.
        start_payload, stop_payload = i * 80, (i + 1) * 80
        payload[start_payload:stop_payload] = 64 * np.concatenate((ifft_output[48:], ifft_output))

    return payload

def ifft128_gi(symbol_stream, start_symbol_index=0):
    """
    Versão do transmissor que usa uma IFFT de 128 pontos para fazer o upsampling,
    em vez de um filtro de meia banda. É mais intensivo em hardware, mas evita
    o "smearing" (borramento) nas bordas dos símbolos.
    Referência: Livro-texto, Seção 7.2.8.
    """
    num_symbols = len(symbol_stream) // 48
    payload = np.zeros(num_symbols * 160, dtype=complex)
    ifft64_input = np.zeros(64, dtype=complex)
    ifft128_input = np.zeros(128, dtype=complex)

    for i in range(num_symbols):
        symbol_idx = start_symbol_index + i

        start_symbol, stop_symbol = i * 48, (i + 1) * 48
        ifft64_input[:] = 0
        ifft64_input[DATA_CARRIERS_IDX] = symbol_stream[start_symbol:stop_symbol]
        ifft64_input[PILOT_CARRIERS_IDX] = PILOT_BASE_POLARITY * PILOT_POLARITY[symbol_idx % 127]

        ifft128_input[:] = 0
        # Mapeia as 64 subportadoras nas posições correspondentes da IFFT de 128 pontos.
        ifft128_input[np.concatenate((np.arange(32), np.arange(96, 128)))] = ifft64_input

        ifft_output = np.fft.ifft(ifft128_input)
        # O GI agora tem 32 amostras para manter a duração do símbolo (4us a 40 MS/s).
        start_payload, stop_payload = i * 160, (i + 1) * 160
        payload[start_payload:stop_payload] = 128 * np.concatenate((ifft_output[96:], ifft_output))

    return payload
