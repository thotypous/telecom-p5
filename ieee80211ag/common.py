# =============================================================================
# Constantes Globais e Funções Comuns ao RX e ao TX
# =============================================================================

import numpy as np

# LUTs de constelação em ordem Gray, com normalização de potência média para 1
# conforme IEEE 802.11a, Seção 17.3.5.7 e Tabela 81. O gr-ieee802-11 define
# os mesmos fatores em `lib/constellations_impl.cc`.

# BPSK: 1 bit por símbolo.
BPSK_LUT = np.array([-1, 1])

# QPSK: 1 bit por eixo, normalizado por sqrt(2).
QPSK_LUT = np.array([-1, 1]) / np.sqrt(2)

# 16-QAM: níveis {-3, -1, 1, 3}; E[I^2 + Q^2] = 10.
QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)

# 64-QAM: níveis {-7, -5, -3, -1, 1, 3, 5, 7}; E[I^2 + Q^2] = 42.
QAM64_LUT = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)

# Sequência de polaridade dos pilotos conforme IEEE 802.11a (Seção 17.3.5.9)
# Esta sequência de 127 elementos evita linhas espectrais nos pilotos. O elemento
# p_0 é usado no SIGNAL; p_1 em diante, nos símbolos DATA. O gr-ieee802-11
# define a mesma tabela em `lib/equalizer/base.cc` como `POLARITY`.
PILOT_POLARITY = np.array([
    1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1,
   -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
   -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1,
   -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1,
   -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
   -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
   -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1
])

# Índices dos pilotos na numeração relativa ao DC (-26 a +26).
# Referência: IEEE 802.11a, Seção 17.3.5.8; livro-texto, Figura 7-15.
PILOT_CARRIERS = np.array([-21, -7, 7, 21])

# Índices das subportadoras piloto no vetor de entrada da IFFT de 64 pontos.
# Para k < 0, o índice FFT é 64 + k. Na ordem fftshift usada por
# `lib/frame_equalizer_impl.cc`, esses pilotos aparecem em 11, 25, 39 e 53.
PILOT_CARRIERS_IDX = np.array([64-21, 64-7, 7, 21])

# Polaridade base dos quatro pilotos antes da modulação pela sequência pseudoaleatória.
# Ordem: subportadoras -21, -7, 7 e 21. A polaridade final também inclui p_n.
PILOT_BASE_POLARITY = np.array([1, 1, 1, -1])

# Índices das 48 subportadoras de dados no vetor de entrada da IFFT de 64 pontos.
# A norma usa 52 tons ativos: 48 de dados e 4 pilotos. A lista abaixo contém
# os tons de dados, excluindo DC e pilotos, já convertidos para índices FFT.
DATA_CARRIERS_IDX = np.array([
    # Mapeamento para subportadoras negativas (-26 a -1, excluindo -21, -7)
    # Índice FFT = 64 + k_subportadora
    38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63,
    # Mapeamento para subportadoras positivas (1 a 26, excluindo 7, 21)
    # Índice FFT = k_subportadora
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26
])

# Tabela de mapeamento para as taxas de dados definidas na norma IEEE 802.11a.
# Cada taxa define a modulação (N_BPSC), taxa de código, bits codificados (N_CBPS)
# e bits de dados (N_DBPS) por símbolo OFDM.
# Referência: Norma IEEE 802.11a-1999, Tabela 78 (Página 17).
RATE_MAP = {
    0b1101: {'name': 'BPSK 1/2',  'Mbps': 6,  'n_bpsc': 1, 'n_dbps': 24,  'n_cbps': 48},
    0b1111: {'name': 'BPSK 3/4',  'Mbps': 9,  'n_bpsc': 1, 'n_dbps': 36,  'n_cbps': 48},
    0b0101: {'name': 'QPSK 1/2',  'Mbps': 12, 'n_bpsc': 2, 'n_dbps': 48,  'n_cbps': 96},
    0b0111: {'name': 'QPSK 3/4',  'Mbps': 18, 'n_bpsc': 2, 'n_dbps': 72,  'n_cbps': 96},
    0b1001: {'name': '16-QAM 1/2','Mbps': 24, 'n_bpsc': 4, 'n_dbps': 96,  'n_cbps': 192},
    0b1011: {'name': '16-QAM 3/4','Mbps': 36, 'n_bpsc': 4, 'n_dbps': 144, 'n_cbps': 192},
    0b0001: {'name': '64-QAM 2/3','Mbps': 48, 'n_bpsc': 6, 'n_dbps': 192, 'n_cbps': 288},
    0b0011: {'name': '64-QAM 3/4','Mbps': 54, 'n_bpsc': 6, 'n_dbps': 216, 'n_cbps': 288},
}

def get_short_training_sequence(step):
    """
    Gera a forma de onda no domínio do tempo para a Sequência de Treinamento Curta (STS).

    A STS ocupa os primeiros 8 us do preâmbulo. Suas 12 subportadoras não nulas
    são espaçadas por 4 tons, criando periodicidade de 16 amostras a 20 MS/s.
    Essa periodicidade alimenta a detecção de pacote, AGC e estimativa grosseira
    de frequência.

    Referências: livro-texto, Seção 7.2.3; IEEE 802.11a, Seção 17.3.3 e Tabela G.2.
    """
    # Valores complexos das 12 subportadoras ativas da STS.
    positive = np.array([0,0,0,0, -1-1j, 0,0,0, -1-1j, 0,0,0,  1+1j, 0,0,0,
                         1+1j, 0,0,0,  1+1j, 0,0,0,  1+1j, 0,0,0, 0,0,0,0])
    negative = np.array([0,0,0,0, 0,0,0,0,  1+1j, 0,0,0, -1-1j, 0,0,0,
                         1+1j, 0,0,0, -1-1j, 0,0,0, -1-1j, 0,0,0,  1+1j, 0,0,0])

    # O fator de escala sqrt(13/6) normaliza a potência média do símbolo resultante.
    # A norma o especifica para garantir que a potência da STS seja consistente com
    # o resto do pacote.
    total = np.sqrt(13/6) * np.concatenate((negative, positive))
    m = np.arange(-32, 32)

    N = 64  # A norma especifica uma IFFT de 64 pontos.
    # A STS completa tem 10 repetições de 16 amostras a 20 MS/s.
    num_samples = int(160 / step)
    short_training_sequence = np.zeros(num_samples, dtype=complex)

    # IDFT manual para permitir `step=0.5` na geração a 40 MS/s.
    for n in range(num_samples):
        t = n * step # Mapeia o índice da amostra para um "tempo" normalizado
        E = np.exp(1j * 2 * np.pi * t * m / N)
        short_training_sequence[n] = np.dot(total, E.T)

    return short_training_sequence

def get_long_training_sequence(step):
    """
    Gera a sequência de treinamento longa (Long Training Sequence - LTS) do preâmbulo 802.11a.

    A LTS fornece os tons conhecidos usados para estimativa de canal e
    sincronização fina de frequência. A sequência no domínio da frequência
    corresponde ao vetor `LONG` de `lib/equalizer/base.cc` no gr-ieee802-11.

    Referências: livro-texto, Seção 7.2.3; IEEE 802.11a, Seção 17.3.3.
    """
    # Valores BPSK das subportadoras da LTS no domínio da frequência.
    positive = np.array([0, 1,-1,-1, 1, 1,-1, 1, -1, 1,-1,-1, -1,-1,-1, 1,
                         1,-1,-1, 1, -1, 1,-1, 1,  1, 1, 1, 0,  0, 0, 0, 0])
    negative = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1,-1, 1, 1, -1, 1,-1, 1,
                         1, 1, 1, 1, 1,-1,-1, 1,  1,-1, 1,-1,  1, 1, 1, 1])

    # Ordem de frequência: tons negativos seguidos dos positivos.
    all_tones = np.concatenate((negative, positive))
    m = np.arange(-32, 32)

    N = 64 # Tamanho da IFFT
    num_samples = int(64 / step) # O número de amostras depende da taxa (step=1 para 20MHz, 0.5 para 40MHz)
    long_training_symbol = np.zeros(num_samples, dtype=complex)

    # IDFT manual para permitir `step=0.5` na geração a 40 MS/s.
    for n in range(num_samples):
        t = n * step # Mapeia o índice da amostra para um "tempo" normalizado
        E = np.exp(1j * 2 * np.pi * t * m / N)
        long_training_symbol[n] = np.dot(all_tones, E.T)

    # LTS completa: GI2 seguido por duas repetições do símbolo longo.
    if step == 1: # Caso de 20 MS/s (64 amostras por símbolo)
        # O GI2 é composto pelas últimas 32 amostras do símbolo
        long_training_sequence = np.concatenate((long_training_symbol[-32:],
                                                 long_training_symbol,
                                                 long_training_symbol))
    else: # Caso de 40 MS/s (128 amostras por símbolo)
        # O GI2 é composto pelas últimas 64 amostras do símbolo
        long_training_sequence = np.concatenate((long_training_symbol[-64:],
                                                 long_training_symbol,
                                                 long_training_symbol))

    return long_training_sequence, all_tones

def scramble(data_bits, initial_state=0b1011101):
    """
    Embaralha os dados de entrada usando uma sequência pseudoaleatória.

    O scrambler quebra longas sequências de bits iguais, reduzindo viés DC e
    concentração espectral. A norma especifica o registrador de 7 bits com
    polinômio S(x) = x^7 + x^4 + 1.

    Referências: livro-texto, Seção 5.6; IEEE 802.11a, Seção 17.3.5.4;
    gr-ieee802-11, `lib/utils.cc`, função `scramble`.
    """
    scrambled = np.zeros_like(data_bits)
    # O estado inicial deve ser não nulo.
    state = np.array([int(b) for b in format(initial_state, '07b')], dtype=int)
    for i in range(len(data_bits)):
        # Feedback do polinômio x^7 + x^4 + 1.
        feedback = state[6] ^ state[3]
        scrambled[i] = data_bits[i] ^ feedback
        state[1:] = state[:-1]
        state[0] = feedback
    return scrambled
