# =============================================================================
# Funções do Receptor (RX)
# =============================================================================

import numpy as np
import binascii
import logging

from .bcc import decode_soft
from .common import *

_DEINTERLEAVING_PATTERNS = {} # Cache para os padrões

def packet_detector(rx_input):
    """
    Detecta a presença de um pacote 802.11a e fornece uma estimativa de temporização grosseira.

    Args:
        rx_input: Amostras complexas I/Q a 20 MS/s.

    Returns:
        Uma tupla `(comparison_ratio, packet_det_flag, falling_edge_position,
        auto_corr_est)`. A borda de descida é usada pelas próximas etapas como
        referência grosseira do fim da STS e início da região da LTS.

    A STS do preâmbulo se repete a cada 16 amostras a 20 MS/s. Por isso, a
    autocorrelação com atraso 16 forma um platô quando há um pacote presente.
    A métrica usada para decisão é normalizada pela potência local para que o
    limiar dependa menos da potência recebida.

    Convenções: os limiares 0.85/0.65 implementam histerese; o limite `i < 1000`
    evita aceitar uma borda tardia como início de pacote nesta prática.

    Referências: livro-texto, Seção 7.3.2; IEEE 802.11a, Seção 17.3.3;
    gr-ieee802-11, `lib/sync_short.cc`.
    """
    detection_flag = 0  # Flag de estado interno para a lógica de histerese
    # Buffers para armazenar os resultados em cada instante de tempo
    auto_corr_est = np.zeros(len(rx_input), dtype=complex)
    comparison_ratio = np.zeros(len(rx_input))
    packet_det_flag = np.zeros(len(rx_input))

    # Buffers que atuam como registradores de deslocamento para os filtros de média móvel
    delay16 = np.zeros(16, dtype=complex)  # Armazena as últimas 16 amostras para o atraso da autocorrelação
    sliding_average1 = np.zeros(32, dtype=complex) # Buffer para a média da autocorrelação
    sliding_average2 = np.zeros(32, dtype=float)   # Buffer para a média da potência (variância)
    falling_edge_position = -1 # Posição da borda de descida, nossa referência de tempo grosseira. -1 = não encontrado.

    for i in range(len(rx_input)):
        rx_input_16 = delay16[15]
        delay16[1:] = delay16[:15]
        delay16[0] = rx_input[i]

        # TAREFA DO ALUNO: calcule a autocorrelação instantânea, atualize a
        # média móvel de autocorrelação e armazene `auto_corr_est[i]`.

        # TAREFA DO ALUNO: calcule a potência instantânea, atualize a média móvel
        # de potência e use-a para normalizar `comparison_ratio[i]`.

        # TAREFA DO ALUNO: aplique a histerese 0.85/0.65 para atualizar
        # `detection_flag` e grave o resultado em `packet_det_flag[i]`.
        packet_det_flag[i] = detection_flag

        # TAREFA DO ALUNO: detecte a transição de 1 para 0 de `packet_det_flag`
        # dentro da janela inicial do pacote e atualize `falling_edge_position`.

    return comparison_ratio, packet_det_flag, falling_edge_position, auto_corr_est

def detect_frequency_offsets(rx_input, falling_edge_position):
    """
    Estima os deslocamentos de frequência grosseiro e fino usando a fase da
    autocorrelação das sequências de treinamento curta e longa, respectivamente.

    Args:
        rx_input: Amostras complexas I/Q a 20 MS/s.
        falling_edge_position: Referência grosseira produzida por
            `packet_detector`.

    Returns:
        Array de dois valores em Hertz: estimativa grosseira pela STS e
        estimativa fina pela LTS.

    A fórmula aplicada nas duas etapas é `theta = 2*pi*Delta_f*(D/Fs)`, em que
    `D` é o atraso da autocorrelação: 16 amostras para a STS e 64 para a LTS.
    O sinal já deve estar aproximadamente alinhado no tempo; por isso os pontos
    de medição são escolhidos em regiões estáveis das sequências de treinamento.

    Referências: livro-texto, Seção 7.3.3; IEEE 802.11a, Seção 17.3.3;
    gr-ieee802-11, `lib/sync_short.cc` e `lib/sync_long.cc`.
    """
    frequency_offsets = np.zeros(2)

    # 1. Cálculo do Deslocamento Grosseiro (usando sequência curta, período 16)
    auto_corr_est = np.zeros(len(rx_input), dtype=complex)
    delay16 = np.zeros(16, dtype=complex)
    sliding_average1 = np.zeros(32, dtype=complex)

    # TAREFA DO ALUNO: calcule a autocorrelação com atraso 16, meça a fase em
    # uma posição estável da STS e converta essa fase para Hertz. Uma escolha
    # robusta é medir cerca de 50 amostras antes de `falling_edge_position`, onde
    # a autocorrelação ainda está no platô da sequência curta. Verifique os
    # limites do vetor antes de acessar essa posição.

    # 2. Cálculo do Deslocamento Fino (usando sequência longa, período 64)
    auto_corr_est_fine = np.zeros(len(rx_input), dtype=complex)
    delay64 = np.zeros(64, dtype=complex)
    sliding_average2 = np.zeros(64, dtype=complex)

    # TAREFA DO ALUNO: repita o procedimento para a LTS, agora com atraso 64,
    # e grave a estimativa fina em `frequency_offsets[1]`. Para os sinais desta
    # prática, uma posição cerca de 125 amostras depois de `falling_edge_position`
    # cai em uma região estável das duas repetições longas.

    return frequency_offsets

def long_symbol_correlator(long_training_symbol, rx_waveform, falling_edge_position):
    """
    Realiza uma correlação cruzada entre a forma de onda recebida e a sequência de
    treinamento longa conhecida para encontrar a posição exata do símbolo de treinamento.

    Args:
        long_training_symbol: Um símbolo LTS local de 64 amostras.
        rx_waveform: Amostras complexas I/Q recebidas.
        falling_edge_position: Referência grosseira obtida pela STS.

    Returns:
        Tupla `(lt_peak_value1, lt_peak_position, output_long)`, contendo o valor
        do maior pico na janela esperada, sua posição e o vetor de correlação.

    Esta versão usa a LTS quantizada pelo sinal das componentes I/Q, como no
    livro-texto. Esse detalhe não aparece nos slides: ele troca coeficientes
    complexos de ponto flutuante por valores em {-1, 0, 1} + j{-1, 0, 1},
    reduzindo o custo de multiplicação do correlacionador. O `gr-ieee802-11`
    usa coeficientes completos em `lib/sync_long.cc`, priorizando a correlação
    exata e contando com filtros/VOLK do GNU Radio.
    """

    # Quasi-cross-correlation: preserva o padrão de fase/sinal da LTS, mas
    # descarta amplitudes intermediárias. Isso é suficiente para produzir picos
    # de sincronização fortes, embora não maximize a saída de correlação como uma
    # LTS completa faria.
    L = np.sign(np.real(long_training_symbol)) + 1j * np.sign(np.imag(long_training_symbol))

    output_long = np.zeros(len(rx_waveform), dtype=complex)
    lt_peak_value1 = 0
    lt_peak_position = 0

    cross_correlator = np.zeros(64, dtype=complex)

    # Taps de um correlacionador FIR: sequência invertida e conjugada. O `flip`
    # transforma a convolução implementada pelo filtro em correlação; o `conj`
    # é necessário porque os símbolos da LTS são complexos.
    L_flipped_conj = np.conj(np.flip(L))

    # TAREFA DO ALUNO: implemente o correlacionador deslizante e procure o maior
    # pico apenas na janela plausível depois da borda grosseira. Use a estrutura
    # temporal do preâmbulo: a LTS aparece depois do GI2, e uma janela de 64
    # amostras começando aproximadamente em `falling_edge_position + 54` evita
    # falsos picos fora da região esperada.

    return lt_peak_value1, lt_peak_position, output_long

def ofdm_receiver(rx_waveform_20mhz, sample_advance, correct_frequency_offset,
                  number_of_ofdm_symbols, use_max_ratio_combining):
    """
    Implementa o receptor OFDM 802.11a completo.

    Args:
        rx_waveform_20mhz: Amostras I/Q do pacote na taxa de 20 MS/s.
        sample_advance: Número de amostras para adiantar a janela da FFT para
            dentro do prefixo cíclico.
        correct_frequency_offset: Habilita ou desabilita as correções grosseira
            e fina de frequência.
        number_of_ofdm_symbols: Limite máximo de símbolos DATA a processar.
        use_max_ratio_combining: Usa pesos proporcionais à magnitude dos pilotos
            na estimativa de fase quando diferente de zero.

    Returns:
        Vetor com os símbolos de dados equalizados e corrigidos, em blocos de
        48 subportadoras por símbolo OFDM. Retorna array vazio quando a
        sincronização falha.

    Convenções: o vetor de 64 subportadoras está na ordem natural da FFT do
    NumPy; `DATA_CARRIERS_IDX` e `PILOT_CARRIERS_IDX` fazem a conversão entre
    subportadoras 802.11 e índices do array.

    Referências: livro-texto, Seções 7.3.1, 7.3.5 e 7.3.6; gr-ieee802-11,
    `examples/wifi_phy_hier.grc`, `lib/equalizer/ls.cc` e
    `lib/frame_equalizer_impl.cc`.
    """
    # 1. Detecta a STS e usa sua borda de descida como referência grosseira
    # para a LTS.
    _, _, falling_edge_position, _ = packet_detector(rx_waveform_20mhz)
    logging.info(f"Falling Edge Position: {falling_edge_position}")
    if falling_edge_position > 600 or falling_edge_position < 0:
        logging.info(f"Error in Falling Edge Position: {falling_edge_position}")
        return np.array([])

    # 2. Corrige o deslocamento de frequência grosseiro medido na STS.
    freq_offset = detect_frequency_offsets(rx_waveform_20mhz, falling_edge_position)
    coarse_offset = freq_offset[0]
    if correct_frequency_offset == 1:
        n = np.arange(len(rx_waveform_20mhz))

        # Mistura digital por -coarse_offset para cancelar a rotação medida.
        # O `gr-ieee802-11` aplica a etapa análoga em `lib/sync_short.cc`.
        nco_signal = np.exp(-1j * 2 * np.pi * n * coarse_offset / 20e6)
        rx_waveform_20mhz *= nco_signal

    # 3. Repete a estimativa na LTS para a correção fina.
    freq_offset = detect_frequency_offsets(rx_waveform_20mhz, falling_edge_position)
    fine_offset = freq_offset[1]
    if correct_frequency_offset == 1:
        n = np.arange(len(rx_waveform_20mhz))
        nco_signal = np.exp(-1j * 2 * np.pi * n * fine_offset / 20e6)
        rx_waveform_20mhz *= nco_signal

    # 4. Sincronização fina de tempo pela correlação com a LTS local.
    long_training_sequence, all_tones = get_long_training_sequence(1)

    # Usa um único símbolo longo como molde do correlacionador.
    long_training_symbol = long_training_sequence[32:96]

    _, lt_peak_position, _ = long_symbol_correlator(long_training_symbol, rx_waveform_20mhz, falling_edge_position)

    # Adianta a janela da FFT para dentro do prefixo cíclico, reduzindo risco
    # de ISI por multipercurso sem perder ortogonalidade.
    lt_peak_position -= sample_advance

    # 5. Estimativa de canal e configuração do equalizador.
    if lt_peak_position < 64 or lt_peak_position + 64 > len(rx_waveform_20mhz):
        print("LTPeak_Position out of bounds for symbol extraction.")
        return np.array([])

    first_long_symbol = rx_waveform_20mhz[lt_peak_position - 64 : lt_peak_position]
    second_long_symbol = rx_waveform_20mhz[lt_peak_position : lt_peak_position + 64]
    averaged_long_training_symbol = first_long_symbol * 0.5 + second_long_symbol * 0.5

    fft_of_long_training_symbol = (1/64) * np.fft.fft(averaged_long_training_symbol)
    all_tones_fft_order = np.array(np.fft.fftshift(all_tones), dtype=float)

    # A LTS tem tons nulos em DC e nas bandas de guarda. Eles não participam da
    # equalização de dados, mas entram no vetor de 64 posições; usar um epsilon
    # evita warning de divisão por zero sem afetar os tons de dados/pilotos.
    all_tones_fft_order[all_tones_fft_order == 0] = 1e-9

    # Nos tons conhecidos da LTS, H ~= Y/X. O equalizador zero-forcing usa 1/H.
    # Referências: livro-texto, Seções 7.1.3.5 e 7.3.5; gr-ieee802-11,
    # `lib/equalizer/ls.cc`.
    channel_estimate = fft_of_long_training_symbol / all_tones_fft_order
    equalizer_coefficients = 1 / channel_estimate

    # 6. Calcula pesos para combinar os pilotos na estimativa de fase.
    pilot_strength = np.abs(channel_estimate[PILOT_CARRIERS_IDX])
    sum_it = np.sum(pilot_strength)
    if use_max_ratio_combining == 0 or sum_it == 0:
        # Usa pesos iguais para reproduzir a estimativa por soma vetorial dos
        # pilotos. Isso é próximo ao `gr-ieee802-11`, em que
        # `lib/frame_equalizer_impl.cc` calcula `beta = arg(...)` sobre a soma dos
        # pilotos já derotados pela polaridade esperada. Não há divisão por 4 no C++
        # porque `arg()` é invariante à escala da soma.
        mrc_coef = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        # Quando habilitado, aproxima uma combinação ponderada por confiabilidade:
        # pilotos com maior magnitude na estimativa de canal recebem maior peso.
        # Esta variação é inspirada no princípio de MRC discutido no livro-texto,
        # Seção 8.1.3, mas não é exigida pela norma IEEE 802.11a.
        mrc_coef = pilot_strength / sum_it

    # 7. Processa todos os símbolos recebidos.
    #
    # `L` controla a memória dos laços lentos de fase e temporização. O símbolo
    # atual é corrigido imediatamente, mas apenas 1/L da correção é incorporada
    # ao equalizador para os próximos símbolos. Isso evita que uma medição ruidosa
    # de piloto desloque todos os coeficientes de uma vez.
    L = 8
    average_slope_filter = np.zeros(L)
    corrected_symbols = np.zeros(48 * number_of_ofdm_symbols, dtype=complex)

    for i in range(number_of_ofdm_symbols):
        start = (i * 80) + lt_peak_position + 64 + 16
        stop = start + 64

        if stop > len(rx_waveform_20mhz):
            logging.info(f"Símbolo {i+1} fora dos limites. Interrompendo o processamento.")
            corrected_symbols = corrected_symbols[:i*48]  # Trunca para os símbolos processados
            break

        current_ofdm_symbol = rx_waveform_20mhz[start:stop]
        current_fft_output = (1/64) * np.fft.fft(current_ofdm_symbol)
        equalized_symbol = current_fft_output * equalizer_coefficients

        # Remove a polaridade esperada dos pilotos antes de estimar fase. A ordem
        # precisa permanecer a mesma de `PILOT_CARRIERS`: -21, -7, 7, 21.
        # TAREFA DO ALUNO: extraia os pilotos equalizados e remova a polaridade
        # conhecida do símbolo OFDM atual.
        pilots = np.ones(4, dtype=complex)

        # Soma vetorial ponderada dos pilotos derotados. Com pesos iguais, isso
        # reproduz a estimativa de fase por soma usada pelo `gr-ieee802-11`; com
        # pesos de canal, pilotos mais confiáveis têm maior influência.
        # TAREFA DO ALUNO: combine os quatro pilotos usando `mrc_coef`.
        averaged_pilot = 1 + 0j

        # Erro de fase comum do símbolo. Referências: livro-texto, Seção 7.3.6;
        # gr-ieee802-11, variável `beta` em `lib/frame_equalizer_impl.cc`.
        # Este ângulo é comum a todas as subportadoras do símbolo atual.
        # TAREFA DO ALUNO: estime o ângulo do piloto médio.
        theta = 0.0

        # Derotação comum da constelação.
        derotation_scalar = np.exp(-1j * theta)
        corrected_symbol1 = equalized_symbol * derotation_scalar

        # Integra lentamente a correção comum no equalizador para suavizar o
        # rastreamento de fase entre símbolos. Esta etapa não é exigida pela
        # norma; é uma escolha de implementação para evitar que o erro residual
        # volte a crescer logo após a derotação do símbolo atual.
        equalizer_coefficients *= np.exp(-1j * theta / L)

        # Erro de temporização residual aparece como inclinação de fase versus
        # frequência. Dividir a fase de cada piloto pelo índice da subportadora
        # transforma a fase medida em estimativa de inclinação. A simetria dos
        # pilotos em torno de DC ajuda a cancelar o componente de fase comum.
        # TAREFA DO ALUNO: estime a inclinação média de fase dos pilotos.
        slope = 0.0

        # Calcula a média da inclinação de fase (slope) ao longo dos últimos `L` símbolos
        # OFDM. Isso suaviza a estimativa, tornando-a mais robusta ao ruído.
        # Um erro de temporização muda muito lentamente, então fazer a média de várias
        # estimativas consecutivas nos dá um resultado mais estável.
        average_slope_filter[1:] = average_slope_filter[:-1]
        average_slope_filter[0] = slope
        average_slope = np.sum(average_slope_filter) / min(i+1, L)

        # Correção de fase residual entre portadoras.
        k = np.fft.fftshift(np.arange(-32, 32))
        applied_correction = k * average_slope

        # Remove a rampa de fase causada pelo erro de temporização. Um atraso
        # dentro do prefixo cíclico não destrói a ortogonalidade, mas aparece
        # como fase linear em k; por isso a correção depende do índice da
        # subportadora e não é o mesmo escalar usado para `theta`.
        corrected_symbol2 = corrected_symbol1 * np.exp(-1j * applied_correction)

        # Suaviza a correção de temporização ao longo dos próximos símbolos.
        # Em um receptor completo, essa informação poderia também dirigir um
        # interpolador de temporização; aqui ela é absorvida no equalizador.
        equalizer_coefficients *= np.exp(-1j * applied_correction / L)

        start_corr = i * 48
        stop_corr = start_corr + 48

        corrected_symbols[start_corr:stop_corr] = corrected_symbol2[DATA_CARRIERS_IDX]

    return corrected_symbols

def decode_signal_field(signal_symbol_iq: np.ndarray) -> dict:
    """
    Decodifica o símbolo SIGNAL (48 símbolos complexos) de um quadro 802.11a/g.

    Este símbolo contém informações cruciais sobre o restante do pacote, como
    a taxa de transmissão e o comprimento dos dados. A decodificação segue os
    passos inversos da transmissão, conforme a norma IEEE 802.11a:
    1. Demodulação BPSK "soft" para obter Log-Likelihood Ratios (LLRs).
    2. De-interleaving dos LLRs.
    3. Decodificação convolucional Viterbi com soft-decision.

    Referências: IEEE 802.11a, Seção 17.3.4; livro-texto, Seções 7.2.4
    e 7.3.1; gr-ieee802-11, `lib/frame_equalizer_impl.cc`.
    Referência interna: inverso de `create_signal_field()` em `ieee80211ag/tx.py`.

    Args:
        signal_symbol_iq: Um array NumPy de 48 números complexos, representando
                          as subportadoras de dados do primeiro símbolo OFDM
                          após a equalização de canal.

    Returns:
        Um dicionário contendo os parâmetros decodificados:
        - 'rate_info': Dicionário com a taxa de transmissão (string e Mbps).
        - 'length': O comprimento do PSDU (MAC frame) em bytes (inteiro).
        - 'parity_ok': Booleano indicando se a verificação de paridade passou.
        - 'tail_ok': Booleano indicando se os 6 bits de cauda são zero.
        - 'raw_bits': Os 24 bits decodificados como um array NumPy.
    """
    # --- PASSO 1: Demodulação BPSK Soft ---
    # O SIGNAL é sempre BPSK com BCC taxa 1/2. Antes dele ser decodificado,
    # o receptor ainda não conhece a modulação nem o comprimento do DATA.
    soft_bits = demapper_ofdm(signal_symbol_iq, 1)

    # --- PASSO 2: De-interleaving ---
    # Padrão de de-interleaving para um símbolo OFDM (48 bits).
    deinterleaving_pattern = create_deinterleaving_pattern(48, 1)
    # Reordena os bits soft para a ordem original, de antes do entrelaçamento.
    deinterleaved_soft_bits = soft_bits[deinterleaving_pattern]

    # --- PASSO 3: Decodificação Viterbi ---
    decoded_bits = convolutional_decoder(deinterleaved_soft_bits)

    # --- PASSO 4: Parsing dos 24 bits decodificados ---
    params = {}
    params['raw_bits'] = decoded_bits

    # Os bits são transmitidos LSB primeiro nos campos RATE e LENGTH.
    # Precisamos inverter a ordem para a conversão para inteiro.

    # Bits 0-3: RATE
    rate_bits = decoded_bits[0:4]
    rate_val = int("".join(map(str, rate_bits)), 2)
    params['rate_info'] = RATE_MAP.get(rate_val, {'name': 'Invalid Rate', 'Mbps': 0})

    # Bit 4: Reservado (deve ser 0)

    # Bits 5-16: LENGTH (12 bits)
    length_bits = decoded_bits[5:17]
    params['length'] = int("".join(map(str, length_bits[::-1])), 2)

    # Bit 17: Paridade par para os primeiros 17 bits (0-16)
    parity_calc = np.sum(decoded_bits[0:17]) % 2
    parity_recv = decoded_bits[17]
    params['parity_ok'] = (parity_calc == parity_recv)

    # Bits 18-23: Cauda (devem ser 0)
    params['tail_ok'] = not np.any(decoded_bits[18:24])

    return params

def decode_data_symbols(data_symbols_iq: np.ndarray, rate_info: dict, psdu_length: int) -> tuple:
    """
    Decodifica os símbolos de dados OFDM (após o campo SIGNAL) de um quadro 802.11a.

    Esta cadeia usa RATE e LENGTH já extraídos do SIGNAL para desfazer o caminho
    de bits do transmissor: demapper soft, deinterleaver, Viterbi, descrambler e
    verificação de CRC.

    Referências: IEEE 802.11a, Seções 17.3.5.4 a 17.3.5.7; livro-texto,
    Seções 7.2.4 e 7.3.1; gr-ieee802-11, `lib/decode_mac.cc`.
    Referência interna: inverso de `encode_data_field()` em `ieee80211ag/tx.py`.

    Args:
        data_symbols_iq: Array NumPy de N x 48 símbolos complexos equalizados.
        rate_info: Dicionário contendo os parâmetros da taxa de transmissão
                   (e.g., {'name': 'QPSK 1/2', 'Mbps': 12}).
        psdu_length: O comprimento do PSDU em bytes, obtido do campo SIGNAL.

    Returns:
        Uma tupla (psdu_bytes, crc_ok):
        - psdu_bytes: Os bytes do quadro MAC (PSDU) recuperado.
        - crc_ok: Booleano indicando se a verificação de CRC32 passou.
    """
    # --- Parâmetros baseados na taxa de transmissão ---
    n_bpsc = rate_info['n_bpsc']
    n_cbps = rate_info['n_cbps']

    # --- PASSO 1: Demodulação Soft (símbolos para LLRs) ---
    soft_bits = demapper_ofdm(data_symbols_iq, n_bpsc)

    # --- PASSO 2: De-interleaving ---
    num_ofdm_symbols = len(data_symbols_iq) // 48
    soft_bits_matrix = soft_bits.reshape(num_ofdm_symbols, n_cbps)

    deinterleave_pattern = create_deinterleaving_pattern(n_cbps, n_bpsc)

    deinterleaved_matrix = soft_bits_matrix[:, deinterleave_pattern]
    deinterleaved_soft_bits = deinterleaved_matrix.flatten()

    # --- PASSO 3: Decodificação Viterbi ---
    # O número total de bits codificados = SERVICE (16) + PSDU (8*len) + TAIL (6)
    total_data_bits = 16 + psdu_length * 8 + 6
    # O número total de bits codificados será o dobro para a taxa 1/2
    total_coded_bits = total_data_bits * 2

    # Trunca a entrada para o Viterbi para o tamanho esperado
    viterbi_input = deinterleaved_soft_bits[:total_coded_bits]

    # Depuncturing poderia ser adicionado aqui para taxas de código > 1/2

    # Executa o algoritmo de Viterbi
    decoded_scrambled_bits = convolutional_decoder(viterbi_input)
    tail_ok = not np.any(decoded_scrambled_bits[-6:])

    # --- PASSO 4: Descrambling ---
    descrambled_bits = descramble(decoded_scrambled_bits)

    # --- PASSO 5: Extração do PSDU e Verificação de CRC ---
    # O fluxo desembaralhado contém: [SERVICE (16 bits)][PSDU (N*8 bits)][TAIL (6 bits)]
    service_field = descrambled_bits[:16]
    psdu_with_crc_bits = descrambled_bits[16 : 16 + psdu_length * 8]

    # Converte bits em bytes
    psdu_with_crc_bytes = np.packbits(psdu_with_crc_bits, bitorder='little')

    # Separa os dados do CRC
    mac_frame_bytes = psdu_with_crc_bytes[:-4]
    received_crc_bytes = psdu_with_crc_bytes[-4:]

    # O CRC32 do 802.11 é padrão, mas o resultado é complementado
    calculated_crc = binascii.crc32(mac_frame_bytes) & 0xFFFFFFFF

    # Converte o CRC recebido (little-endian) para um inteiro
    received_crc = int.from_bytes(received_crc_bytes, byteorder='little')

    crc_ok = (calculated_crc == received_crc)

    return mac_frame_bytes, tail_ok, crc_ok

def descramble(scrambled_bits):
    """
    Desembaralha bits que foram embaralhados pelo scrambler do IEEE 802.11a/g.

    Args:
        scrambled_bits: Bits decodificados pelo Viterbi, ainda embaralhados.

    Returns:
        Bits desembaralhados, com os sete primeiros bits do SERVICE restaurados
        para zero.

    No campo DATA, os primeiros bits do SERVICE são zero antes do scrambling;
    por isso os 7 primeiros bits decodificados revelam o estado inicial usado
    para desembaralhar o restante.

    Convenção: o scrambler do IEEE 802.11 é aditivo em GF(2). Aplicar a mesma
    sequência pseudoaleatória novamente reverte o XOR.

    Referências: IEEE 802.11a, Seção 17.3.5.4; gr-ieee802-11,
    `lib/decode_mac.cc`, função `descramble`.
    """
    scrambler_register = scrambled_bits[:7]
    # Os sete bits aparecem na ordem de transmissão; o estado interno usado por
    # `scramble` deve ser reconstruído com o bit mais recente na posição adequada.
    # TAREFA DO ALUNO: recupere o estado inicial e aplique `scramble` aos bits
    # restantes para desfazer o XOR do embaralhador. A função `scramble` interpreta
    # `initial_state` com `format(initial_state, '07b')`; portanto, alinhe os sete
    # bits recebidos com essa ordem antes de convertê-los para inteiro.
    return np.zeros_like(scrambled_bits)

def convolutional_decoder(soft_bits):
    """
    Decodifica soft bits do código convolucional IEEE 802.11 taxa 1/2, K=7.

    Args:
        soft_bits: Vetor de LLRs aproximados, dois valores por bit de informação.

    Returns:
        Bits estimados pelo Viterbi soft-decision.

    Este wrapper preserva a interface do receptor. O algoritmo de Viterbi em si
    fica em `ieee80211ag/bcc.py`, na função `_decode_soft_numba`, para que o
    núcleo compilado com Numba possa ser estudado e preenchido separadamente.

    Referências: IEEE 802.11a, Seção 17.3.5.5; livro-texto, Seções 5.6.2
    e 5.6.4; gr-ieee802-11, `lib/viterbi_decoder/base.h` e
    `lib/viterbi_decoder/viterbi_decoder_generic.cc`.
    """
    return np.array(decode_soft(soft_bits), dtype=int)

def create_deinterleaving_pattern(n_cbps, n_bpsc):
    """
    Cria e armazena em cache o padrão de de-interleaving para um dado
    número de bits codificados por símbolo (N_CBPS) e por subportadora (N_BPSC).

    Args:
        n_cbps: Bits codificados por símbolo OFDM.
        n_bpsc: Bits codificados por subportadora.

    Returns:
        Vetor de índices que reordena os bits recebidos para a ordem anterior ao
        interleaving do transmissor.

    Convenção: a norma define duas permutações no transmissor. Aqui calculamos a
    permutação inversa para aplicar diretamente sobre cada linha da matriz de
    soft bits recebidos.

    Referências: IEEE 802.11a, Seção 17.3.5.6; gr-ieee802-11,
    `lib/frame_equalizer_impl.cc` e `lib/decode_mac.cc`.
    Referência interna: inverso de `create_interleaving_pattern()` em
    `ieee80211ag/tx.py`.
    """
    if (n_cbps, n_bpsc) in _DEINTERLEAVING_PATTERNS:
        return _DEINTERLEAVING_PATTERNS[(n_cbps, n_bpsc)]

    # Primeira permutação da norma, expressa em função do índice original `k`.
    # TAREFA DO ALUNO: implemente as duas permutações do interleaver e combine-as
    # para obter a permutação inversa aplicada no receptor. Compare com
    # `create_interleaving_pattern()` em `ieee80211ag/tx.py`: lá, `np.argsort`
    # transforma o mapa direto no padrão usado por `bits[pattern]`; aqui, queremos
    # o padrão que desfaz essa reordenação quando aplicado aos soft bits recebidos.
    deinterleave_pattern = np.arange(n_cbps)

    _DEINTERLEAVING_PATTERNS[(n_cbps, n_bpsc)] = deinterleave_pattern
    return deinterleave_pattern

def demapper_ofdm(symbols_iq, n_bpsc):
    """
    Mapeia uma sequência de símbolos de constelação para soft bits (LLRs).

    Args:
        symbols_iq: Símbolos complexos equalizados.
        n_bpsc: Número de bits por subportadora, conforme a taxa do quadro.

    Returns:
        Vetor de soft bits na ordem esperada pelo deinterleaver.

    Para BPSK e QPSK, as componentes real e imaginária são proporcionais aos
    LLRs. Multiplicar todos os LLRs por uma constante positiva não muda o caminho
    escolhido pelo Viterbi; por isso esta implementação não estima a variância de
    ruído. 16-QAM e 64-QAM exigem as expressões completas de LLR e não são
    implementadas aqui. A convenção de sinal é a mesma do mapeamento IEEE:
    valores positivos favorecem bit 1.

    Referências: livro-texto, Seção 5.6.2; IEEE 802.11a, Seção 17.3.5.7.
    """
    if n_bpsc == 1:
        # TAREFA DO ALUNO: para BPSK, retorne a componente real dos símbolos.
        return np.zeros(len(symbols_iq), dtype=float)
    if n_bpsc == 2:
        # TAREFA DO ALUNO: para QPSK, intercale componentes I e Q na ordem
        # [I0, Q0, I1, Q1, ...]. `np.column_stack(...).ravel()` é uma forma
        # compacta de transformar dois vetores I/Q em um único fluxo intercalado.
        return np.zeros(2 * len(symbols_iq), dtype=float)
    raise ValueError(f"n_bpsc={n_bpsc} not supported")
