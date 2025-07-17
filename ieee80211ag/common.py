# =============================================================================
# Constantes Globais e Funções Comuns ao RX e ao TX
# =============================================================================

import numpy as np

# --- Tabelas de Consulta (Look-Up Tables - LUTs) para Mapeamento de Constelação ---
# Estas constantes definem os níveis de amplitude para cada eixo (I ou Q) nas
# diferentes modulações. O design é importante para o desempenho do sistema:
# 1. Mapeamento Gray: Os valores são ordenados de forma a corresponder a um
#    mapeamento Gray, minimizando erros de bit adjacentes.
# 2. Normalização de Potência: Cada LUT é normalizada por um fator específico.
#    Isso garante que a potência média de cada constelação resultante (BPSK, QPSK,
#    16-QAM, 64-QAM) seja sempre igual a 1. Isso é vital para que o transmissor
#    possa mudar de modulação sem precisar reajustar a potência de saída de RF.
#
# - Referência ao Livro-texto: Seção 5.2.1, Figura 5-14 (Página 310) e a discussão
#   sobre a normalização para manter a variância constante na página 309.
# - Referência à Norma IEEE 802.11a: Seção 17.3.5.7 e Tabela 81 (Página 27), que
#   especificam os fatores de normalização exatos (K_MOD).
# - Referência gr-ieee802-11: `lib/constellations_impl.cc`, onde estas mesmas
#   constelações e fatores de normalização são definidos em C++.

# BPSK (Binary Phase Shift Keying): 1 bit por eixo.
# Mapeia 1 bit para os pontos -1 ou +1. A potência média já é 1, então não há normalização.
BPSK_LUT = np.array([-1, 1])

# QPSK (Quadrature Phase Shift Keying): 1 bit por eixo (2 bits por símbolo).
# Usa os mesmos pontos base do BPSK, mas a constelação 2D resultante (pontos em
# ±0.707 ±j0.707) é normalizada por sqrt(2) para que a potência média seja 1.
QPSK_LUT = np.array([-1, 1]) / np.sqrt(2)

# 16-QAM (Quadrature Amplitude Modulation): 2 bits por eixo (4 bits por símbolo).
# Usa 4 níveis de amplitude por eixo: -3, -1, +1, +3.
# O fator de normalização, sqrt(10), é a raiz quadrada da potência média dos 16
# pontos da constelação não normalizada (i.e., E[I² + Q²] = 10).
QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)

# 64-QAM (Quadrature Amplitude Modulation): 3 bits por eixo (6 bits por símbolo).
# Usa 8 níveis de amplitude por eixo: -7, -5, -3, -1, +1, +3, +5, +7.
# O fator de normalização, sqrt(42), é a raiz quadrada da potência média dos 64
# pontos da constelação não normalizada (i.e., E[I² + Q²] = 42).
QAM64_LUT = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)

# Sequência de polaridade dos pilotos conforme IEEE 802.11a (Seção 17.3.5.9)
# A norma 802.11a (Seção 17.3.5.8) especifica que os quatro tons piloto em um símbolo OFDM
# devem ser modulados por BPSK por uma sequência pseudoaleatória para evitar a geração
# de linhas espectrais discretas no sinal transmitido, o que poderia causar interferência.
# Esta sequência p_n, de 127 elementos, é usada ciclicamente. O elemento p_0 é para o
# campo SIGNAL e p_1 em diante para os símbolos de DADOS.
# Referência gr-ieee802-11: `lib/equalizer/base.h` define `POLARITY`.
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

# Índices das subportadoras piloto conforme a numeração relativa ao DC (-26 a +26).
# A norma IEEE 802.11a especifica que quatro subportadoras em cada símbolo OFDM de dados
# são reservadas para pilotos. Estes pilotos são usados pelo receptor para rastrear
# variações de fase e frequência, como as causadas por ruído de fase ou
# deriva do oscilador.
#
# Referência Norma IEEE 802.11a: Seção 17.3.5.8, "Pilot subcarriers". A norma
#   especifica: "pilot signals shall be put in subcarriers –21, –7, 7 and 21."
# Referência Livro-texto: Capítulo 7, Seção 7.2.4, Figura 7-15, que ilustra a
#   distribuição das subportadoras de informação e piloto.
PILOT_CARRIERS = np.array([-21, -7, 7, 21])

# Índices das subportadoras piloto no vetor de entrada da IFFT de 64 pontos.
# A IFFT espera um vetor onde as frequências negativas são mapeadas para a parte
# superior do array. A conversão de um índice de subportadora `k` para um índice de
# IFFT `m` é `m = k` para `k >= 0` e `m = N + k` para `k < 0`, com N=64.
# Assim, -21 se torna 64-21=43 e -7 se torna 64-7=57.
#
# Referência Norma IEEE 802.11a: Seção 17.3.5.9 e Figura 109, que mostram o
#   mapeamento das subportadoras para as entradas da IFFT.
# Referência Livro-texto: Seção 7.2.5, "The Mapping Process" e Figura 7-30,
#   que ilustram o mapeamento de símbolos para a grade da IFFT.
# Referência gr-ieee802-11: O código em `lib/frame_equalizer_impl.cc` acessa os
#   pilotos pelos seus índices de FFT, que são 7, 21, 43, e 57.
PILOT_CARRIERS_IDX = np.array([64-21, 64-7, 7, 21])

# Polaridade base dos quatro pilotos antes da modulação pela sequência pseudoaleatória.
# Estes são os valores BPSK base para os pilotos nas subportadoras -21, -7, 7 e 21,
# respectivamente. Cada um desses valores é então multiplicado pelo elemento `p_n` da
# sequência de polaridade (PILOT_POLARITY) para o símbolo OFDM `n` atual.
#
# Referência Norma IEEE 802.11a: Seção 17.3.5.9. Embora a polaridade final dependa
#   da sequência `p_n`, esta sequência base é um detalhe de implementação comum
#   para gerar os símbolos de treinamento longos e os pilotos de dados.
# Referência gr-ieee802-11: O código em `frame_equalizer_impl.cc` usa esta
#   polaridade base implicitamente ao corrigir os pilotos, por exemplo,
#   `pilot[3] = in[53] * -p;` onde o `-p` indica a polaridade base -1 para
#   o piloto em SC=+21 (índice 53, pois o gnuradio usa convenção fftshift).
PILOT_BASE_POLARITY = np.array([1, 1, 1, -1])

# Índices das 48 subportadoras de dados no vetor de entrada da IFFT de 64 pontos.
# A norma 802.11a usa 52 subportadoras no total (-26 a 26, excluindo 0), das quais
# 4 são para pilotos. As 48 restantes são para dados. Esta constante lista os
# índices da IFFT para essas 48 subportadoras de dados na ordem correta em que
# os bits são mapeados para as subportadoras.
#
# Referência Norma IEEE 802.11a: Seção 17.3.5.7, "Subcarrier modulation mapping".
#   A norma especifica que as subportadoras -26 a -1 e 1 a 26 são usadas, com
#   exceção das subportadoras piloto, resultando em 48 subportadoras de dados.
# Referência Livro-texto: Figura 7-15, que mostra visualmente as subportadoras
#   de dados, piloto, e as nulas (DC e banda de guarda).
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

    A STS é a primeira parte do preâmbulo 802.11a (dura 8 µs) e é usada pelo
    receptor para múltiplas tarefas de sincronização críticas:
    1.  **Detecção de Pacote:** Sinaliza o início de um quadro.
    2.  **Controle Automático de Ganho (AGC):** Permite que o receptor ajuste seu ganho
        para o nível de potência do sinal recebido.
    3.  **Sincronização de Frequência Grosseira:** Fornece uma primeira estimativa do
        deslocamento de frequência entre transmissor e receptor.

    Para alcançar isso, a STS é construída a partir de apenas 12 subportadoras
    não-nulas, espaçadas por 4 posições no domínio da frequência. Esta
    estrutura esparsa no domínio da frequência cria uma forma de onda periódica no
    domínio do tempo, com um período de 16 amostras (a 20 MS/s), que é a
    propriedade explorada pelo detector de pacotes.

    - Referência ao Livro-texto: Seção 7.2.3, "The Short Training Sequence" (Página 506).
      A Figura 7-20 mostra a resposta de frequência desta sequência.
    - Referência à Norma IEEE 802.11a: Seção 17.3.3 e Tabela G.2 (página 64)
      definem os valores exatos das subportadoras.
    """
    # --- Definição das Subportadoras no Domínio da Frequência ---
    # Define os valores complexos (QPSK) para as 12 subportadoras ativas.
    # A maioria das subportadoras é zerada. Esta esparsidade (apenas as subportadoras
    # com índice múltiplo de 4 são usadas) é o que cria a periodicidade de 16
    # amostras (64/4=16) no domínio do tempo.
    positive = np.array([0,0,0,0, -1-1j, 0,0,0, -1-1j, 0,0,0,  1+1j, 0,0,0,
                         1+1j, 0,0,0,  1+1j, 0,0,0,  1+1j, 0,0,0, 0,0,0,0])
    negative = np.array([0,0,0,0, 0,0,0,0,  1+1j, 0,0,0, -1-1j, 0,0,0,
                         1+1j, 0,0,0, -1-1j, 0,0,0, -1-1j, 0,0,0,  1+1j, 0,0,0])

    # O fator de escala sqrt(13/6) normaliza a potência média do símbolo resultante.
    # A norma o especifica para garantir que a potência da STS seja consistente com
    # o resto do pacote.
    total = np.sqrt(13/6) * np.concatenate((negative, positive))
    m = np.arange(-32, 32)

    # --- Geração da Forma de Onda no Domínio do Tempo ---
    N = 64  # A norma especifica uma IFFT de 64 pontos.
    # A STS completa tem 10 repetições do símbolo base de 0.8 µs (16 amostras a 20 MS/s),
    # totalizando 160 amostras (8 µs). O `step` ajusta para taxas de amostragem
    # mais altas (e.g., 40 MS/s).
    num_samples = int(160 / step)
    short_training_sequence = np.zeros(num_samples, dtype=complex)

    # Este loop implementa manualmente a Transformada Discreta de Fourier Inversa (IDFT).
    # Isso é feito para ter controle explícito sobre a taxa de amostragem através do `step`.
    # x[n] = Σ X[m] * e^(j*2*π*n*m/N)
    for n in range(num_samples):
        t = n * step # Mapeia o índice da amostra para um "tempo" normalizado
        # Calcula as exponenciais complexas (base da transformada de Fourier) para este instante t
        E = np.exp(1j * 2 * np.pi * t * m / N)
        # O produto escalar realiza a soma ponderada, gerando a amostra no domínio do tempo
        short_training_sequence[n] = np.dot(total, E.T)

    return short_training_sequence

def get_long_training_sequence(step):
    """
    Gera a sequência de treinamento longa (Long Training Sequence - LTS) do preâmbulo 802.11a.

    Esta função constrói a forma de onda de 8 µs no domínio do tempo para a LTS, que
    serve a dois propósitos críticos no receptor:
    1.  **Estimativa de Canal Fina:** Fornece um sinal de referência conhecido para que o
        receptor possa medir as distorções de amplitude e fase introduzidas pelo
        canal de múltiplos percursos.
    2.  **Sincronização Fina de Frequência:** A estrutura repetitiva da LTS permite
        ao receptor refinar a estimativa de deslocamento de frequência obtida
        na etapa anterior.

    - Referência ao Livro-texto: Seção 7.2.3, "The Long Training Sequence" (Página 508).
    - Referência à Norma IEEE 802.11a: Seção 17.3.3, que define a sequência `L` e a
      estrutura da LTS.
    - Referência gr-ieee802-11: A sequência de frequência (`all_tones`) é hardcoded como
      o vetor estático `LONG` no arquivo `lib/equalizer/base.h`.
    """
    # Define os valores BPSK (+1, -1, 0) para as subportadoras no domínio da frequência,
    # conforme especificado na norma. O vetor é dividido em metades positiva e negativa
    # para corresponder ao formato de entrada da IFFT.
    positive = np.array([0, 1,-1,-1, 1, 1,-1, 1, -1, 1,-1,-1, -1,-1,-1, 1,
                         1,-1,-1, 1, -1, 1,-1, 1,  1, 1, 1, 0,  0, 0, 0, 0])
    negative = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1,-1, 1, 1, -1, 1,-1, 1,
                         1, 1, 1, 1, 1,-1,-1, 1,  1,-1, 1,-1,  1, 1, 1, 1])

    # Concatena as metades para formar o vetor de entrada completo de 64 pontos para a IFFT.
    # A ordem (negativa primeiro) é a convenção padrão para a FFT/IFFT em muitas bibliotecas,
    # representando as frequências de -N/2 a N/2-1.
    all_tones = np.concatenate((negative, positive))
    m = np.arange(-32, 32)

    # --- Geração do Símbolo de Treinamento Longo no Domínio do Tempo via IDFT ---
    N = 64 # Tamanho da IFFT
    num_samples = int(64 / step) # O número de amostras depende da taxa (step=1 para 20MHz, 0.5 para 40MHz)
    long_training_symbol = np.zeros(num_samples, dtype=complex)

    # Este loop implementa manualmente a Transformada Discreta de Fourier Inversa (IDFT).
    # Isso é feito para ter controle explícito sobre a taxa de amostragem através do `step`.
    # x[n] = Σ X[m] * e^(j*2*π*n*m/N)
    for n in range(num_samples):
        t = n * step # Mapeia o índice da amostra para um "tempo" normalizado
        # Calcula as exponenciais complexas (base da transformada de Fourier) para este instante t
        E = np.exp(1j * 2 * np.pi * t * m / N)
        # O produto escalar realiza a soma ponderada, gerando a amostra no domínio do tempo
        long_training_symbol[n] = np.dot(all_tones, E.T)

    # --- Construção da Sequência de Treinamento Longa Completa ---
    # A LTS completa no ar não é apenas um símbolo, mas sim um Intervalo de Guarda (GI2)
    # seguido por DUAS repetições idênticas do símbolo de treinamento longo (T1 e T2).
    # O GI2 é um prefixo cíclico especial, com metade do comprimento do símbolo,
    # para garantir uma transição suave do preâmbulo curto.
    # A repetição de T1 e T2 permite que o receptor faça uma média, reduzindo o ruído
    # e melhorando a precisão da estimativa de canal.
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

    Esta etapa é fundamental para garantir o bom desempenho do transmissor e do receptor.
    O objetivo é aleatorizar a sequência de bits de dados, quebrando longas sequências
    de '0's ou '1's. Ao contrário do que se poderia pensar em sistemas de portadora única,
    em OFDM o objetivo principal *não* é a recuperação de clock a partir dos dados. Em vez disso,
    o scrambling serve para:

    1.  **Evitar Viés DC e Tons Espectrais Fortes:** Longas sequências de bits idênticos
        podem levar a símbolos de constelação repetitivos. Isso pode criar um forte
        componente DC (frequência zero) ou picos de potência em subportadoras específicas
        após a IFFT. Isso é altamente indesejável, pois pode saturar amplificadores e
        interferir com os circuitos sensíveis a DC de um receptor de conversão direta (Zero-IF).
    2.  **Garantir um Espectro de Potência Plano ("Whitening"):** Ao tornar os dados
        estatisticamente aleatórios, garantimos que a potência do sinal seja distribuída
        de forma mais uniforme por todas as subportadoras. Isso ajuda a atender aos
        requisitos da máscara espectral e torna o sinal mais robusto.
    3.  **Auxiliar a Estabilidade de Subsistemas do Receptor:** Embora não dirija a
        recuperação de clock, um sinal com estatísticas semelhantes às do ruído (espectro plano,
        sem DC) ajuda no funcionamento estável de outros laços, como o Controle
        Automático de Ganho (AGC). Um AGC estável é um pré-requisito para que os
        algoritmos de sincronização subsequentes funcionem corretamente.

    - Referência ao Livro-texto: Seção 5.6, "The Transmit Bit Chain" (Página 356). O
      livro se refere a esta etapa como "Whitening (Scrambling)" e explica seu
      propósito em evitar longas sequências de bits e concentração de potência.
    - Referência à Norma IEEE 802.11a: A Seção 17.3.5.4 ("PLCP DATA scrambler and
      descrambler") e a Figura 113 especificam este exato registrador de deslocamento
      com o polinômio gerador S(x) = x⁷ + x⁴ + 1.
    - Referência gr-ieee802-11: `lib/utils.cc`, função `scramble`. A implementação em
      C++ realiza a mesma operação bit a bit.
    """
    scrambled = np.zeros_like(data_bits)
    # O estado do scrambler é um registrador de deslocamento de 7 bits. O estado inicial
    # é pseudoaleatório e não-nulo, conforme a norma.
    state = np.array([int(b) for b in format(initial_state, '07b')], dtype=int)
    for i in range(len(data_bits)):
        # O bit de realimentação (feedback) é gerado pelo XOR dos bits na 4ª e 7ª posição
        # do registrador de deslocamento, correspondendo ao polinômio x⁷ + x⁴ + 1.
        # No array `state` com índice 0 a 6, isso corresponde a state[3] e state[6].
        feedback = state[6] ^ state[3]
        # O bit de saída é o XOR do bit de entrada com o bit de feedback.
        scrambled[i] = data_bits[i] ^ feedback
        # Desloca o registrador para a direita, inserindo o bit de feedback na primeira posição.
        state[1:] = state[:-1]
        state[0] = feedback
    return scrambled
