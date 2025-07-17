# =============================================================================
# Funções do Receptor (RX)
# =============================================================================

import numpy as np
from viterbi import Viterbi
import binascii
import logging

from .common import *

_DEINTERLEAVING_PATTERNS = {} # Cache para os padrões

def packet_detector(rx_input):
    """
    Detecta a presença de um pacote 802.11a e fornece uma estimativa de temporização grosseira.

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    Esta função implementa um detector de pacotes robusto que explora a estrutura
    periódica da Sequência de Treinamento Curta (Short Training Sequence) do preâmbulo 802.11a.
    O método principal é a autocorrelação: comparar o sinal recebido com uma versão
    atrasada de si mesmo. Como a sequência curta se repete a cada 16 amostras (a 20 MS/s),
    um pico na autocorrelação com atraso de 16 amostras indica a provável
    presença de um pacote.

    A função deve retornar não apenas a decisão, mas também uma referência de tempo
    crítica (`falling_edge_position`) que sincroniza todas as etapas subsequentes do receptor.

    --------------------------------------------------------------------------
    ## Lógica de Implementação:

    Você precisará iterar sobre cada amostra de `rx_input`, mantendo o estado em
    buffers de atraso para simular um processamento em tempo real.

    1.  **Inicialize as Variáveis:**
        - Crie os arrays de saída (`auto_corr_est`, `comparison_ratio`, `packet_det_flag`)
          preenchidos com zeros e com o mesmo tamanho de `rx_input`.
        - Crie os buffers de atraso para as médias móveis: `delay16` (tamanho 16, complexo),
          `sliding_average1` (tamanho 32, complexo, para autocorrelação) e
          `sliding_average2` (tamanho 32, float, para potência/variância).
        - Inicialize `falling_edge_position = -1`.

    2.  **Loop Principal (para cada amostra `i` em `rx_input`):**
        a. **Simule o Atraso de 16 Amostras:**
           - Salve a amostra mais antiga de `delay16`.
           - Desloque o buffer `delay16` em uma posição.
           - Insira a amostra atual `rx_input[i]` na primeira posição de `delay16`.

        b. **Calcule a Autocorrelação Suavizada:**
           - Calcule a autocorrelação instantânea: `temp = rx_input[i] * np.conj(rx_input_16)`.
           - Use o buffer `sliding_average1` como um filtro de média móvel de 32 pontos
             para suavizar `temp`, produzindo `auto_corr_est[i]`.

        c. **Calcule a Variância Suavizada:**
           - Calcule a potência instantânea: `inst_power = np.abs(rx_input[i])**2`.
           - Use o buffer `sliding_average2` como um filtro de média móvel de 32 pontos
             para suavizar `inst_power`, produzindo `variance_est`.

        d. **Calcule a Razão de Comparação Normalizada:**
           - Divida a magnitude da autocorrelação suavizada (`abs_auto_corr_est`)
             pela variância suavizada (`variance_est`) para obter `comparison_ratio[i]`.
             Esta normalização torna a detecção robusta a variações de potência do sinal.

        e. **Aplique a Lógica de Detecção com Histerese:**
           - Se `comparison_ratio[i]` for maior que um limiar superior (e.g., 0.85),
             defina o estado de detecção como 1 (pacote detectado).
           - Se `comparison_ratio[i]` for menor que um limiar inferior (e.g., 0.65),
             defina o estado de detecção como 0 (sem pacote).
           - Armazene o estado atual em `packet_det_flag[i]`.

        f. **Capture a Borda de Descida:**
           - Verifique se houve uma transição no `packet_det_flag` de 1 para 0.
           - Se isso ocorrer (e estiver dentro de uma janela de tempo razoável,
             e.g., `i < 1000`), armazene o índice `i` atual em `falling_edge_position`.
             Esta será a única referência de tempo que você precisa guardar.

    3.  **Retorne os Resultados:** Retorne `comparison_ratio`, `packet_det_flag`,
        `falling_edge_position` e `auto_corr_est`.

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Livro-texto:** Seção 7.3.2, "Packet Detection" (Página 523) e
      especialmente a **Figura 7-40**, que mostra o diagrama de blocos exato deste
      algoritmo. A **Figura 7-41** ilustra a interpretação gráfica dos resultados.
    - **Norma IEEE 802.11a:** Seção 17.3.3, que define a estrutura e
      a periodicidade de 0.8 µs (16 amostras a 20 MS/s) da sequência de treinamento
      curta, que é a propriedade explorada aqui.
    - **gr-ieee802-11:** `lib/sync_short.cc`. O bloco C++ implementa um
      conceito análogo, procurando por um "platô" de valores de correlação acima
      de um limiar (`threshold`) para detectar o pacote. A lógica de histerese
      neste código Python serve ao mesmo propósito de estabilizar a detecção.
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

    # SEU CÓDIGO AQUI

    return comparison_ratio, packet_det_flag, falling_edge_position, auto_corr_est

def detect_frequency_offsets(rx_input, falling_edge_position):
    """
    Estima os deslocamentos de frequência grosseiro e fino em um sinal 802.11a.

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    Esta função implementa um estimador de deslocamento de frequência de dois estágios.
    O deslocamento de frequência ocorre porque os osciladores de cristal do
    transmissor e do receptor nunca operam exatamente na mesma frequência. Este
    erro, se não for corrigido, causa uma rotação contínua da constelação,
    tornando a demodulação impossível.

    -   **Estimativa Grosseira:** Usa a Sequência de Treinamento Curta (STS) para estimar
        e corrigir grandes deslocamentos de frequência. Tem uma faixa de captura ampla,
        mas baixa precisão.
    -   **Estimativa Fina:** Usa a Sequência de Treinamento Longa (LTS) para refinar a
        estimativa, corrigindo o erro residual com alta precisão.

    Ambas as estimativas exploram o mesmo princípio: um deslocamento de frequência `Δf`
    causa uma rotação de fase de `2π * Δf * T` ao longo de um período de tempo `T`.
    Ao medir essa rotação de fase na autocorrelação de uma sequência periódica,
    podemos calcular `Δf`.

    --------------------------------------------------------------------------
    ## Lógica de Implementação:

    A função deve ser dividida em duas seções principais, uma para cada estimativa.

    **PARTE 1: Cálculo do Deslocamento de Frequência Grosseiro**

    1.  **Calcule a Autocorrelação com Atraso de 16 Amostras:**
        -   Assim como na função `packet_detector`, você precisa calcular a
            autocorrelação suavizada do `rx_input` com um atraso de 16 amostras.
            Reutilize a mesma lógica de loop com `delay16` e `sliding_average1`
            para gerar o vetor `auto_corr_est`.

    2.  **Selecione o Ponto de Medição:**
        -   A referência de tempo `falling_edge_position` marca o fim da STS.
            Escolha um ponto de medição estável *dentro* da STS, recuando
            um número fixo de amostras (e.g., 50) a partir desta referência.
            Isso garante que o AGC do receptor já tenha convergido.

    3.  **Extraia a Fase e Calcule a Frequência:**
        -   No ponto de medição escolhido (`theta_pos`), extraia o ângulo de fase
            do valor complexo da autocorrelação (`theta = np.angle(...)`).
        -   Converta este ângulo de fase (em radianos) para uma frequência em Hertz
            usando a fórmula: `freq_offset = theta * Fs / (2 * pi * Atraso)`.
            Aqui, `Fs` é a taxa de amostragem (20e6 Hz) e `Atraso` é 16 amostras.
        -   Armazene este resultado como o primeiro elemento do seu array de saída.

    **PARTE 2: Cálculo do Deslocamento de Frequência Fino**

    1.  **Calcule a Autocorrelação com Atraso de 64 Amostras:**
        -   O processo é idêntico ao anterior, mas agora o atraso é de 64 amostras,
            correspondendo à periodicidade das duas metades da LTS.
        -   Use novos buffers (`delay64`, `sliding_average2`) para calcular a
            autocorrelação suavizada `auto_corr_est_fine`.

    2.  **Selecione o Ponto de Medição:**
        -   A LTS começa após a STS. Escolha um ponto de medição no meio da
            segunda metade da LTS para máxima robustez. Um bom ponto é
            `falling_edge_position + 125`.

    3.  **Extraia a Fase e Calcule a Frequência:**
        -   No novo `theta_pos`, extraia o ângulo de fase de `auto_corr_est_fine`.
        -   Use a mesma fórmula de conversão de fase para frequência, mas agora com
            `Atraso` igual a 64. Usar um atraso maior (64 vs. 16) resulta em
            uma estimativa de frequência 4 vezes mais precisa para o mesmo erro
            de medição de fase.
        -   Armazene este resultado como o segundo elemento do seu array de saída.

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Livro-texto:** Seção 7.3.3, "Frequency Offset Detection". A **Figura 7-43
      (Página 527)** é a referência visual e matemática chave, mostrando como o
      ângulo de correlação (`θ_short` ou `θ_long`) é mapeado para um deslocamento de
      frequência.

    - **Norma IEEE 802.11a:** Seção 17.3.3 define as estruturas da STS (com
      período de 16 amostras) e da LTS (com período de 64 amostras), que são
      as propriedades exploradas por esta função.

    - **gr-ieee802-11:** O bloco `sync_short.cc` calcula a estimativa grosseira
      (`d_freq_offset = arg(...) / 16;`), e o `sync_long.cc` calcula a estimativa
      fina (`d_freq_offset = arg(...) / 64;`). Este código Python implementa
      exatamente a mesma lógica, mas converte o resultado final para Hertz.
    """
    frequency_offsets = np.zeros(2)

    # 1. Cálculo do Deslocamento Grosseiro (usando sequência curta, período 16)
    auto_corr_est = np.zeros(len(rx_input), dtype=complex)
    delay16 = np.zeros(16, dtype=complex)
    sliding_average1 = np.zeros(32, dtype=complex)

    # SEU CÓDIGO AQUI

    # 2. Cálculo do Deslocamento Fino (usando sequência longa, período 64)
    #
    # O código é essencialmente análogo ao anterior, alterando-se o período.
    # Enquanto a estimativa grosseira usou a periodicidade de 16 amostras da 
    # sequência curta, a estimativa fina usará a periodicidade de 64 amostras
    # da sequência longa, o que permite uma medição de fase muito mais precisa.
    auto_corr_est_fine = np.zeros(len(rx_input), dtype=complex)
    delay64 = np.zeros(64, dtype=complex)
    sliding_average2 = np.zeros(64, dtype=complex)

    # SEU CÓDIGO AQUI

    return frequency_offsets

def long_symbol_correlator(long_training_symbol, rx_waveform, falling_edge_position):
    """
    Realiza uma correlação cruzada para encontrar a posição exata do símbolo
    de treinamento longo e fornecer uma referência de temporização fina.

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    Após o `packet_detector` nos dar uma localização aproximada do pacote, esta
    função precisa encontrar o ponto exato para o início da FFT com precisão de
    uma única amostra. Ela faz isso deslizando uma cópia local e ideal da Sequência
    de Treinamento Longa (LTS) sobre a forma de onda recebida e encontrando o
    ponto de máxima correlação (o pico).

    --------------------------------------------------------------------------
    ## Lógica de Implementação:

    Você implementará um correlacionador deslizante. A principal otimização aqui é
    o uso de uma "Quasi-Correlação Cruzada", onde os coeficientes do filtro são
    quantizados para +1/-1/+j/-j, eliminando a necessidade de multiplicações complexas.

    1.  **Crie os Coeficientes do Correlacionador (1-bit Quantized):**
        -   Receba o `long_training_symbol` ideal.
        -   Crie um novo vetor de coeficientes `L` onde a parte real é `sign(real(long_training_symbol))`
            e a parte imaginária é `sign(imag(long_training_symbol))`. Isso quantiza
            os coeficientes para os cantos do quadrado unitário no plano complexo.

    2.  **Prepare os Coeficientes para a Convolução:**
        -   Para que uma operação de convolução (que é o que um filtro FIR faz)
            produza uma correlação, o kernel do filtro deve ser a versão
            **conjugada e invertida no tempo** da sequência que você está procurando.
        -   Aplique `np.conj(np.flip(L))` para obter os coeficientes `L_flipped_conj`
            que serão usados no seu filtro.

    3.  **Inicialize os Buffers:**
        -   Crie um buffer `cross_correlator` de 64 amostras (o comprimento da LTS),
            inicializado com zeros. Ele atuará como o registrador de deslocamento do
            seu filtro FIR.
        -   Inicialize variáveis para armazenar o valor do pico (`lt_peak_value1`) e
            sua posição (`lt_peak_position`).

    4.  **Loop Principal (para cada posição `i` em `rx_waveform`):**
        a. **Calcule a Saída da Correlação:**
           - Calcule o produto escalar (`np.dot`) entre o buffer `cross_correlator`
             (que contém as 64 amostras mais recentes do sinal) e os coeficientes
             `L_flipped_conj`. O resultado é o valor da correlação para o instante `i`.

        b. **Atualize o Buffer Deslizante:**
           - Desloque o buffer `cross_correlator` em uma posição, descartando a
             amostra mais antiga.
           - Insira a amostra atual `rx_waveform[i]` na primeira posição do buffer.

        c. **Procure pelo Pico na Janela de Interesse:**
           - Crie uma janela de busca.
           - **Dentro desta janela**, compare a magnitude do `output` da correlação
             atual com a do maior pico encontrado até agora (`lt_peak_value1`).
           - Se o `output` atual for maior, atualize `lt_peak_value1` com o valor
             complexo do `output` e `lt_peak_position` com o índice `i` atual.

    5.  **Retorne os Resultados:** Retorne o valor do pico, a posição do pico e o
        vetor completo de saída da correlação (útil para depuração).

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Livro-texto:**
        - Seção 7.3.4, "Timing Acquisition" (Página 529): Discute a necessidade da
          sincronização fina e introduz o conceito de "QuasiCrossCorrelation"
          para otimização, que é exatamente o que está sendo implementado aqui.
        - Figura 7-45 (Página 529): Mostra o diagrama de blocos de um correlacionador
          deslizante. Sua implementação do loop com o buffer `cross_correlator` e o
          produto escalar seguirá este diagrama.
        - Figura 2-7 (Página 137): Explica por que os coeficientes do correlacionador
          precisam ser invertidos no tempo ("flipped").

    - **Norma IEEE 802.11a:**
        - Seção 17.3.3: Define a LTS, que é o "template" (`long_training_symbol`)
          usado nesta função.

    - **gr-ieee802-11:**
        - `lib/sync_long.cc`: Implementa a sincronização longa. No entanto, ele usa uma
          abordagem diferente baseada na **autocorrelação** da LTS (explorando sua
          estrutura repetitiva T1/T2), enquanto este código Python usa a
          **correlação cruzada** com um template local. Ambas são técnicas válidas
          para alcançar o mesmo objetivo.
    """
    L = np.sign(np.real(long_training_symbol)) + 1j * np.sign(np.imag(long_training_symbol))

    output_long = np.zeros(len(rx_waveform), dtype=complex)
    lt_peak_value1 = 0
    lt_peak_position = 0

    cross_correlator = np.zeros(64, dtype=complex)

    # SEU CÓDIGO AQUI

    return lt_peak_value1, lt_peak_position, output_long

def ofdm_receiver(rx_waveform_20mhz, sample_advance, correct_frequency_offset,
                  number_of_ofdm_symbols, use_max_ratio_combining):
    """
    Implementa o receptor OFDM 802.11a completo.
    Referência: Livro-texto, Seção 7.3, Figura 7-37.
    Referência gr-ieee802-11: `wifi_phy_hier.grc`.
    """
    # 1. Código do Receptor: Decimação por 2
    # Existia no livro-texto, mas aqui fazemos em outra parte do código.
    # rx_waveform_20mhz = tx_output[::2]

    # 2. Detector de Pacote
    #
    # Esta é a primeira etapa ativa do receptor. A função `packet_detector` analisa
    # a forma de onda de entrada (`rx_waveform_20mhz`) para encontrar a assinatura
    # da sequência de treinamento curta, que sinaliza o início de um potencial
    # pacote 802.11a.

    # Executa a função de detecção. Estamos interessados principalmente no
    # `falling_edge_position`, que é a nossa primeira e mais importante estimativa
    # de temporização. Ela nos informa onde a sequência de treinamento curta termina
    # e a sequência longa começa. As outras saídas são descartadas com `_`.
    #
    # - Referência ao Livro-texto: Seção 7.3.1, "Short Training Sequence". O texto
    #   descreve as tarefas de "Packet detection" e "Coarse frequency correction"
    #   que são habilitadas por esta sequência.
    _, _, falling_edge_position, _ = packet_detector(rx_waveform_20mhz)
    logging.info(f"Falling Edge Position: {falling_edge_position}")
    if falling_edge_position > 600 or falling_edge_position < 0:
        logging.info(f"Error in Falling Edge Position: {falling_edge_position}")
        return np.array([])

    # 3. Detectar e Corrigir o Deslocamento de Frequência Grosseiro
    freq_offset = detect_frequency_offsets(rx_waveform_20mhz, falling_edge_position)
    coarse_offset = freq_offset[0]
    if correct_frequency_offset == 1:
        n = np.arange(len(rx_waveform_20mhz))

        # Esta linha gera uma forma de onda senoidal complexa no domínio do tempo discreto.
        # Essa forma de onda é a representação digital de um sinal de oscilador local (LO).
        # Seu propósito é ser multiplicada pelo sinal recebido para anular o deslocamento
        # de frequência grosseiro (coarse frequency offset) estimado anteriormente.
        # Este processo é matematicamente equivalente à operação de "mistura" (mixing)
        # em um receptor analógico para fazer a conversão de frequência para a banda base.
        #
        # - Detalhe da Operação:
        #   - `np.exp(...)`: Utiliza a fórmula de Euler, e^(jθ), para gerar um vetor de
        #     números complexos que traçam um círculo unitário.
        #   - `-1j`: O sinal negativo na exponencial complexa indica que a rotação
        #     será no sentido horário. Isso é crucial, pois estamos gerando um sinal
        #     para *cancelar* um deslocamento de frequência positivo medido. Se o
        #     sinal recebido foi deslocado por +f, nós o multiplicamos por um sinal
        #     de -f para trazê-lo de volta a 0 Hz.
        #   - `2 * np.pi * n * coarse_offset / 20e6`: Este é o termo de fase instantânea θ(n).
        #     - `coarse_offset`: A estimativa do deslocamento de frequência em Hz.
        #     - `20e6`: A taxa de amostragem (Fs) em Hz.
        #     - `coarse_offset / 20e6`: É a frequência normalizada (f/Fs), que representa
        #       a frequência em "ciclos por amostra".
        #     - `n`: É o índice da amostra, representando o tempo discreto.
        #
        # - Referência ao Livro-texto:
        #   - Seção 2.3.1, "The Frequency-shifting Property" (Página 150): Discute
        #     o princípio fundamental de que a multiplicação no domínio do tempo por uma
        #     exponencial complexa `e^(-j2πf₀t)` causa um deslocamento no espectro de
        #     frequência. Esta linha é a implementação discreta dessa propriedade.
        #   - Seção 6.3.5, "Frequency Offset" (Página 461) e Figura 6-41: Descreve o
        #     problema do deslocamento de frequência e como ele é modelado
        #     matematicamente como uma multiplicação por uma exponencial complexa.
        #
        # - Referência à Norma IEEE 802.11a:
        #   - Seção 17.3.3: A estrutura do preâmbulo curto é projetada especificamente
        #     para permitir a estimativa do `coarse_offset`. Este código é a
        #     aplicação dessa estimativa para corrigir o sinal. A norma exige que os
        #     receptores tolerem deslocamentos de frequência de até ±20 ppm, o que
        #     torna esta etapa de correção obrigatória.
        #
        # - Referência gr-ieee802-11:
        #   - No arquivo `lib/sync_short.cc`, o deslocamento de frequência é estimado
        #     e armazenado na variável `d_freq_offset`. A correção é então aplicada
        #     no bloco `sync_long` com a linha `out[o] = in[o] * exp(gr_complex(0, -d_freq_offset * d_copied));`,
        #     que é a contraparte em C++ desta operação em Python.
        nco_signal = np.exp(-1j * 2 * np.pi * n * coarse_offset / 20e6)
        rx_waveform_20mhz *= nco_signal

    # 4. Detectar e Corrigir o Deslocamento de Frequência Fino
    freq_offset = detect_frequency_offsets(rx_waveform_20mhz, falling_edge_position)
    fine_offset = freq_offset[1]
    if correct_frequency_offset == 1:
        n = np.arange(len(rx_waveform_20mhz))
        # Esta linha gera uma segunda senoidal complexa para realizar a correção *fina*
        # do deslocamento de frequência. O princípio de funcionamento e a matemática
        # são idênticos aos da correção grosseira explicada anteriormente.
        #
        # A principal diferença reside na origem e precisão do valor de `fine_offset`:
        # - Ele foi calculado usando a **Sequência de Treinamento Longa**, que, por ser mais
        #   longa, permite uma medição de fase mais precisa e, consequentemente, uma
        #   estimativa de deslocamento de frequência mais acurada.
        # - Esta é a segunda etapa em um processo de duas fases: a correção grosseira
        #   alinha o sinal para dentro da faixa de captura do estimador fino, que
        #   então realiza o ajuste final de alta precisão.
        #
        # - Referência ao Livro-texto: Seção 7.3.3, "Frequency Offset Detection". A seção
        #   descreve o uso de ambas as sequências de treinamento para os estágios grosseiro e fino.
        # - Referência à Norma IEEE 802.11a: Seção 17.3.3, "Long Training Sequence". A norma
        #   especifica que esta sequência é usada para "channel estimation and fine
        #   frequency acquisition".
        # - Referência gr-ieee802-11: O arquivo `lib/sync_long.cc` é responsável por esta
        #   estimativa, calculando o `d_freq_offset` com base na correlação das duas
        #   metades da sequência longa.
        nco_signal = np.exp(-1j * 2 * np.pi * n * fine_offset / 20e6)
        rx_waveform_20mhz *= nco_signal

    # 5. Sincronização Fina de Temporização via Correlação Cruzada
    #
    # Após a detecção do pacote e a correção grosseira de frequência, esta etapa
    # determina o instante de tempo preciso para o início da FFT. A precisão
    # de uma única amostra é crucial aqui, pois um erro pode introduzir um
    # deslocamento de fase que degrada o desempenho. Para isso, usamos a Sequência
    # de Treinamento Longa (LTS), que não possui a estrutura repetitiva da STS
    # e, portanto, fornece um pico de correlação muito mais nítido e preciso.

    # Gera a sequência de treinamento longa ideal (local), incluindo o GI2 e as duas
    # repetições (T1 e T2), para servir como referência. O `step=1` indica que
    # estamos gerando a referência para uma taxa de amostragem de 20 MS/s. `all_tones`
    # contém os valores ideais no domínio da frequência, que serão usados mais tarde
    # para a estimativa do canal.
    # Referência Livro-texto: Seção 7.2.3, "The Long Training Sequence" (Página 508).
    long_training_sequence, all_tones = get_long_training_sequence(1)

    # Extrai apenas um período do símbolo de treinamento longo (64 amostras, ou 3.2 µs),
    # que corresponde a T1 ou T2. Este é o "template" ou "molde" que será usado
    # pelo correlacionador para encontrar a sua contraparte no sinal recebido.
    # Estamos basicamente procurando por "onde está esta forma de onda de 64 amostras
    # dentro do meu sinal recebido?".
    long_training_symbol = long_training_sequence[32:96]

    # Executa o correlacionador deslizante. A função `long_symbol_correlator` compara
    # (correlaciona) o `long_training_symbol` ideal com a `rx_waveform_20mhz` recebida.
    # Ela retorna a posição (`lt_peak_position`) do pico de correlação, que indica
    # o final do primeiro símbolo de treinamento longo (T1) no sinal recebido.
    # Esta posição se torna a nossa referência de temporização de alta precisão (timing reference)
    # para o restante do pacote.
    #
    # - Referência Livro-texto: Seção 7.3.4, "Timing Acquisition" (Página 529) e
    #   Figura 7-45, que ilustra a implementação de um correlacionador deslizante.
    # - Referência Norma IEEE 802.11a: Seção 17.3.3 e Figura 110, que mostram a
    #   estrutura do preâmbulo e a localização dos símbolos de treinamento longos.
    # - Referência gr-ieee802-11: Esta etapa é realizada pelo bloco `sync_long`. O
    #   código em `lib/sync_long.cc` busca por picos de correlação para encontrar
    #   o `d_frame_start`, que é análogo ao `lt_peak_position`.
    _, lt_peak_position, _ = long_symbol_correlator(long_training_symbol, rx_waveform_20mhz, falling_edge_position)

    # A variável `sample_advance` é um parâmetro de ajuste que avança (move para
    # mais cedo no tempo) o ponto de início da FFT em relação à posição do pico
    # de correlação. O pico (`lt_peak_position`) indica o alinhamento perfeito
    # para o *caminho mais forte* do sinal, mas não necessariamente o ponto ideal
    # para iniciar a FFT em um ambiente com múltiplos percursos.
    #
    # O objetivo principal deste avanço é criar uma margem de segurança para evitar a
    # Interferência Intersimbólica (ISI) causada por "pré-cursores" do *próximo*
    # símbolo OFDM. Se um caminho de sinal mais fraco, mas mais rápido, do *próximo*
    # símbolo chegar antes do final do símbolo *atual*, ele pode contaminar
    # a janela da FFT. Ao mover o início da FFT um pouco para dentro do
    # Intervalo de Guarda (GI), reduzimos essa vulnerabilidade.
    #
    # Esta operação é válida porque o GI é um "prefixo cíclico". Mover o início
    # da FFT para dentro do GI não corrompe os dados; apenas introduz um
    # deslocamento de fase linear em todas as subportadoras, que é
    # perfeitamente corrigível pelo equalizador de canal na etapa seguinte.
    #
    # - Referência ao Livro-texto: Este conceito é o foco da Seção 7.3.4
    #   (Página 529) e é ilustrado na **Figura 7-47 (Página 531)**.
    #   A figura mostra a "Preferred Range for FFT Calculation" (Faixa Preferida
    #   para o Cálculo da FFT) começando *antes* do início do "Current IFFT Output",
    #   exatamente para evitar a interferência do "Pre-cursor of Next OFDM Symbol".
    #   O `sample_advance` é a implementação numérica dessa faixa preferida.
    lt_peak_position -= sample_advance

    # 6. Estimativa de Canal e Configuração do Equalizador
    if lt_peak_position < 64 or lt_peak_position + 64 > len(rx_waveform_20mhz):
        print("LTPeak_Position out of bounds for symbol extraction.")
        return np.array([])

    first_long_symbol = rx_waveform_20mhz[lt_peak_position - 64 : lt_peak_position]
    second_long_symbol = rx_waveform_20mhz[lt_peak_position : lt_peak_position + 64]
    averaged_long_training_symbol = first_long_symbol * 0.5 + second_long_symbol * 0.5

    fft_of_long_training_symbol = (1/64) * np.fft.fft(averaged_long_training_symbol)
    all_tones_fft_order = np.array(np.fft.fftshift(all_tones), dtype=float)

    # Evita warning de divisão por zero nos tons nulos
    all_tones_fft_order[all_tones_fft_order == 0] = 1e-9

    # Esta é a etapa fundamental da equalização em OFDM. O princípio é que, para
    # cada subportadora, a saída recebida (Y) é o produto do símbolo transmitido (X)
    # e a resposta em frequência do canal (H) naquela subportadora específica,
    # mais o ruído (N). A relação é Y = H * X + N.
    #
    # Para estimar o canal H, usamos os símbolos de treinamento longos, onde tanto o
    # sinal transmitido (X) quanto o recebido (Y) são conhecidos. Ao ignorar o termo
    # de ruído, podemos estimar H simplesmente dividindo Y por X.
    #
    # - Referência Livro-texto: Seção 7.1.3.5, "Channel Estimation and RS-CINR Estimation"
    #   (Página 489). A fórmula `RawFreqResponse = RefSignal_Rx / RefSignal_Tx`
    #   é a exata representação desta linha de código.
    # - Referência Norma IEEE 802.11a: Seção 17.3.3, "Long Training Sequence". A norma
    #   especifica que esta sequência é usada para "channel estimation", e este
    #   cálculo é a implementação padrão dessa estimativa.
    # - Referência gr-ieee802-11: No arquivo `lib/frame_equalizer_impl.cc`, a estimativa
    #   do canal `d_H` é calculada de forma idêntica, dividindo a FFT do
    #   símbolo de treinamento recebido (`in[i]`) pela sequência ideal conhecida (`LONG[i]`).
    channel_estimate = fft_of_long_training_symbol / all_tones_fft_order
    # Com a estimativa do canal (H) em mãos, podemos construir um equalizador
    # para reverter a distorção do canal. O método mais simples é o "Zero-Forcing" (ZF),
    # que tenta inverter perfeitamente o canal.
    #
    # Se o sinal recebido é Y = H * X, para recuperar X, simplesmente multiplicamos Y
    # pelo inverso do canal, ou seja, X_estimado = Y * (1/H). Portanto, os coeficientes
    # do nosso equalizador são simplesmente o recíproco da estimativa do canal.
    #
    # - Referência Livro-texto: Seção 7.3.5, "Channel Estimation and Equalizer Setup"
    #   (Página 532). A fórmula `Equalizer[m] = Ideal_Tones[m] / RX_Tones[m]` é
    #   matematicamente equivalente a `1 / channel_estimate`. O Capítulo 4, "Estimation,
    #   Equalization and Adaptive Signal Processing", discute a teoria geral dos
    #   equalizadores.
    # - Referência Norma IEEE 802.11a: A norma não dita o tipo de equalizador,
    #   mas a equalização por subportadora no domínio da frequência é um dos
    #   principais benefícios do OFDM, e a equalização ZF é a forma mais direta.
    # - Referência gr-ieee802-11: A equalização é aplicada diretamente em
    #   `lib/frame_equalizer_impl.cc` com a linha `symbols[c] = in[i] / d_H[i];`,
    #   que é a aplicação da multiplicação pelo `equalizer_coefficients` aqui calculado.
    equalizer_coefficients = 1 / channel_estimate

    # 7. Calcula a força do sinal em cada piloto para a Combinação de Máxima Razão (Max Ratio Combining)
    pilot_strength = np.abs(channel_estimate[PILOT_CARRIERS_IDX])
    sum_it = np.sum(pilot_strength)
    if use_max_ratio_combining == 0 or sum_it == 0:
        # O objetivo aqui é calcular os pesos (coeficientes) para a média ponderada
        # dos quatro tons piloto. Em vez de uma média simples onde todos os pilotos
        # teriam o mesmo peso (1/4), usamos a Combinação de Máxima Razão (MRC).
        # A MRC é a técnica ideal para combinar múltiplos sinais em presença de ruído,
        # pois ela maximiza a Relação Sinal-Ruído (SNR) do resultado combinado.
        #
        # A lógica é dar mais peso aos pilotos que foram recebidos com maior
        # potência (maior `pilot_strength`), pois a estimativa de fase deles é mais
        # confiável (menos afetada pelo ruído). Pilotos em "deep fade" (muito
        # atenuados) recebem menos peso, minimizando o impacto de sua fase ruidosa
        # na estimativa final de `theta`.
        #
        # - Detalhe da Operação:
        #   - `pilot_strength`: Um vetor contendo a magnitude (potência) da
        #     estimativa do canal em cada uma das quatro subportadoras piloto.
        #   - `sum_it`: A soma total da potência de todos os pilotos.
        #   - A divisão normaliza os pesos para que a soma deles seja igual a 1.
        #
        # - Referência ao Livro-texto: A teoria da MRC é detalhada no Capítulo 8,
        #   Seção 8.1.3, "Maximum Ratio Combining" (Página 604). Embora o livro
        #   apresente a MRC no contexto de diversidade de antenas, o princípio
        #   matemático de ponderar sinais pela sua força para maximizar a SNR
        #   é exatamente o mesmo que está sendo aplicado aqui aos tons piloto.
        #
        # - Referência à Norma IEEE 802.11a: A norma não exige MRC, mas esta é uma
        #   técnica de implementação padrão da indústria para otimizar o desempenho
        #   do rastreamento de fase, atendendo ao requisito de robustez da Seção 17.3.5.8.
        #
        # - Referência gr-ieee802-11: Curiosamente, a implementação de referência em
        #   `lib/frame_equalizer_impl.cc` usa uma abordagem mais simples chamada
        #   Combinação de Ganho Igual (Equal Gain Combining - EGC), onde os pilotos
        #   são simplesmente somados (`(pilot[0] + pilot[1] + pilot[2] + pilot[3]) / 4`).
        #   A implementação em Python aqui é, portanto, mais otimizada teoricamente
        #   para maximizar a SNR da estimativa de fase.
        mrc_coef = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        mrc_coef = pilot_strength / sum_it

    # 8. Processar todos os Símbolos Recebidos
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

        # Após a equalização inicial, os símbolos ainda podem ter um erro de fase
        # comum a todas as subportadoras. Este erro é causado por ruído de fase do
        # oscilador e qualquer deslocamento de frequência residual que não foi
        # completamente corrigido. Esta etapa usa os quatro tons piloto, que são
        # símbolos BPSK conhecidos inseridos em posições fixas, para medir e
        # corrigir este erro de fase para cada símbolo OFDM individualmente.

        # TAREFA DO ALUNO: Implementar a extração, derotação e média dos pilotos.

        # --- Lógica de Implementação ---

        # 1. Extraia os Símbolos Piloto:
        #    - A partir do `equalized_symbol` (que é um vetor de 64 pontos), use o
        #      índice `PILOT_CARRIERS_IDX` para extrair os 4 símbolos complexos
        #      correspondentes às subportadoras piloto.

        # 2. Derrote os Pilotos para uma Referência Comum:
        #    - A norma 802.11a especifica que a polaridade dos pilotos (+1 ou -1)
        #      muda a cada símbolo OFDM de acordo com uma sequência pseudoaleatória
        #      (armazenada em `PILOT_POLARITY`). Além disso, os pilotos têm uma
        #      polaridade base fixa (`PILOT_BASE_POLARITY`).
        #    - Para medir o erro de fase do *canal*, você precisa primeiro remover a
        #      modulação conhecida dos pilotos. Faça isso multiplicando os pilotos
        #      extraídos pelo conjugado complexo da polaridade total esperada.
        #      Como a polaridade é puramente real (+1 ou -1), a multiplicação é
        #      suficiente. A polaridade esperada para o símbolo `i` é
        #      `PILOT_BASE_POLARITY * PILOT_POLARITY[i % 127]`.
        #    - O resultado desta operação, `pilots`, será um vetor de 4 símbolos
        #      complexos que, idealmente (sem ruído ou erro de fase), seriam todos
        #      iguais a `1+0j`. Qualquer desvio de fase em relação ao eixo real positivo
        #      é o erro de fase do canal.

        pilots = np.array([1+0j, 1+0j, 1+0j, 1+0j])  # SUBSTITUA PELO SEU CÓDIGO

        # 3. Calcule o Piloto Médio Ponderado (MRC):
        #    - Calcule a média ponderada dos 4 pilotos derrotados usando os coeficientes
        #      MRC (`mrc_coef`) que você calculou anteriormente. Isso combina a
        #      informação de fase dos quatro pilotos de forma a maximizar a SNR da
        #      estimativa final. O resultado é `averaged_pilot`.

        averaged_pilot = 1+0j  # SUBSTITUA PELO SEU CÓDIGO

        # 4. Estime o Erro de Fase (Theta):
        #    - Simplesmente calcule o ângulo (em radianos) do `averaged_pilot` complexo.
        #      Este ângulo `theta` é a sua estimativa final e robusta do erro de fase
        #      comum para o símbolo OFDM atual.

        theta = 0  # SUBSTITUA PELO SEU CÓDIGO

        # --- Referências Cruzadas ---

        # - Livro-texto:
        #   - Seção 8.1.3, "Maximum Ratio Combining" (Página 604): Explica a teoria por
        #     trás do uso dos `mrc_coef` para a média ponderada.
        #   - Seção 7.3.6, "Removing Phase and Timing Drift" (Página 534): Descreve
        #     a necessidade e o método de rastreamento de fase usando os pilotos.

        # - Norma IEEE 802.11a:
        #   - Seção 17.3.5.8 ("Pilot subcarriers"): Especifica a existência, localização
        #     e propósito dos pilotos.
        #   - Seção 17.3.5.9 e Equação (25): Definem a sequência de polaridade `p_n`
        #     (`PILOT_POLARITY`) usada para modular os pilotos.

        # - gr-ieee802-11:
        #   - `lib/frame_equalizer_impl.cc`: O cálculo de `beta` é o análogo direto
        #     do cálculo de `theta`. A multiplicação por `p` e `-p` implementa
        #     a derotação da polaridade dos pilotos. Note que o gr-ieee802-11 usa
        #     uma média simples (EGC) em vez de MRC.
        # --------------------------------------------------------------------------

        # Cria um escalar de derotação. Este é um vetor complexo unitário (magnitude 1)
        # com um ângulo de -theta. Multiplicar os símbolos recebidos por este escalar
        # rotaciona toda a constelação em -theta, cancelando o erro de fase medido.
        #
        # - Referência Livro-texto: A correção é a aplicação deste escalar a todos
        #   os símbolos, conforme mencionado na Seção 7.3.6. O uso de exponenciais
        #   complexas para rotação é um conceito fundamental do Capítulo 1.
        # - Referência gr-ieee802-11: Em `lib/frame_equalizer_impl.cc`, a correção é
        #   feita multiplicando `current_symbol` por `exp(gr_complex(0, -beta))`.
        derotation_scalar = np.exp(-1j * theta)
        corrected_symbol1 = equalized_symbol * derotation_scalar

        # Após a correção do símbolo atual, esta linha atualiza os próprios coeficientes
        # do equalizador para rastrear lentamente a deriva de fase (phase drift) ao longo
        # do tempo. Isso é uma forma de "laço de realimentação" (feedback loop) lento.
        #
        # O objetivo é evitar que o erro de fase se acumule ao longo de muitos símbolos
        # (o que aconteceria se apenas corrigíssemos cada símbolo individualmente).
        # Ao ajustar lentamente o próprio equalizador, mantemo-lo "sincronizado" com
        # o estado de fase atual do canal.
        #
        # - Detalhe da Operação:
        #   - `theta`: O erro de fase médio medido para o símbolo *atual*.
        #   - `theta / L`: O erro de fase é dividido por `L`, que é o comprimento
        #     do `AverageSlopeFilter`. Isso age como um fator de suavização ou
        #     "taxa de aprendizado". Estamos aplicando apenas uma pequena fração (1/L)
        #     do erro medido para atualizar os coeficientes. Isso evita que
        #     medições de fase ruidosas em um único símbolo causem grandes
        #     alterações no equalizador. Essencialmente, é uma filtragem passa-baixa
        #     da estimativa de fase do canal.
        #   - `np.exp(-1j * ...)`: Cria um pequeno escalar de rotação corretiva
        #     que é então multiplicado por *todos* os coeficientes do equalizador.
        #
        # - Referência ao Livro-texto: Este é um exemplo prático de um conceito de
        #   **equalização adaptativa**, discutido no Capítulo 4. Embora não seja
        #   uma implementação completa do algoritmo LMS (Seção 4.4), ele compartilha
        #   o princípio fundamental de usar um sinal de erro (`theta`) para
        #   atualizar iterativamente os coeficientes de um filtro (o equalizador).
        #   A Seção 7.3.6 também aborda a necessidade de corrigir a deriva de fase
        #   e de temporização.
        #
        # - Referência à Norma IEEE 802.11a: A norma não especifica o algoritmo
        #   exato para rastreamento de fase, mas exige que o receptor seja robusto
        #   o suficiente para operar em canais com ruído de fase (Seção 17.3.5.8).
        #   Esta linha de código é parte de uma implementação que atende a esse
        #   requisito, usando os tons piloto fornecidos pela norma.
        #
        # - Referência gr-ieee802-11: No código C++ (`lib/equalizer/lms.cc` e `sta.cc`),
        #   uma abordagem conceitualmente semelhante é usada, onde a estimativa do
        #   canal `d_H` é atualizada usando uma média móvel exponencial (um filtro IIR):
        #   `d_H[i] = (1-alpha)*d_H[i] + alpha*H_update[i];`. Ambas as técnicas
        #   (a rotação lenta em Python e o filtro IIR em C++) servem para
        #   suavizar/filtrar a estimativa do canal ao longo do tempo para rastrear
        #   mudanças lentas.
        equalizer_coefficients *= np.exp(-1j * theta / L)

        # Após corrigir o erro de fase *comum* a todas as subportadoras, agora
        # precisamos corrigir o erro de fase que *varia linearmente* com a frequência.
        # Este erro é a assinatura de um deslocamento de temporização (timing offset)
        # residual no domínio do tempo.
        #
        # TAREFA DO ALUNO: Implementar o cálculo da inclinação da fase.
        #
        # --------------------------------------------------------------------------
        # ## Objetivo:
        # O objetivo é medir a inclinação (slope) da reta de fase que passa pelos
        # tons piloto. De acordo com a propriedade de deslocamento no tempo da
        # Transformada de Fourier, um atraso `t₀` no tempo causa uma rampa de fase
        # `φ(f) = -2π * t₀ * f` na frequência. A inclinação desta rampa (`φ(f) / f`)
        # é, portanto, diretamente proporcional ao erro de temporização `t₀`.
        # Ao medir essa inclinação, podemos estimar e corrigir o erro de temporização.
        #
        # --------------------------------------------------------------------------
        # ## Lógica de Implementação:
        #
        # Você calculará uma média ponderada da inclinação de fase para cada piloto.
        #
        # 1.  **Obtenha a Fase dos Pilotos:**
        #     -   Use `np.angle(pilots)` para obter um vetor com o ângulo de fase (φ)
        #         de cada um dos 4 símbolos piloto complexos.
        #
        # 2.  **Obtenha a Frequência dos Pilotos:**
        #     -   Use o vetor de constantes `PILOT_CARRIERS`, que contém os índices
        #         de frequência [-21, -7, 7, 21].
        #
        # 3.  **Calcule a Inclinação Individual:**
        #     -   Divida o vetor de fases pelo vetor de frequências. Esta operação
        #         elemento a elemento (`fase / frequência`) calcula a inclinação
        #         individual para cada piloto.
        #
        # 4.  **Calcule a Média Ponderada (MRC):**
        #     -   Multiplique o vetor de inclinações individuais pelos pesos `mrc_coef`
        #         (calculados anteriormente). Isso dá mais importância aos pilotos
        #         com maior SNR, tornando a estimativa mais robusta.
        #     -   Some os resultados ponderados usando `np.sum()` para obter a
        #         estimativa final e suavizada da inclinação (`slope`).
        #
        # **Nota Importante:** Este cálculo funciona mesmo com um erro de fase comum
        # (`theta`) presente nos pilotos. Devido ao posicionamento simétrico dos pilotos
        # em torno da frequência zero (e.g., -21 e +21), o componente de fase comum
        # é cancelado matematicamente durante a média ponderada, isolando
        # efetivamente a inclinação.
        #
        # --------------------------------------------------------------------------
        # ## Referências Cruzadas:
        #
        # - **Livro-texto:** Seção 7.3.6, "Detecting and Removing Timing Drift"
        #   e a **Figura 7-50 (Página 535)**. A figura ilustra perfeitamente como
        #   um erro de temporização (`timing offset`) cria uma linha reta no
        #   gráfico de fase vs. frequência, cuja inclinação esta linha de código mede.
        #
        # - **Norma IEEE 802.11a:** Seção 17.3.5.8 especifica a existência e a
        #   localização simétrica dos pilotos, que é a propriedade que permite que
        #   este algoritmo funcione de forma robusta.
        #
        # - **gr-ieee802-11:** A implementação em `lib/frame_equalizer_impl.cc`
        #   rastreia a deriva de temporização de forma diferente, calculando o `d_er`,
        #   que é o erro de fase *entre* símbolos consecutivos. O código Python
        #   aqui implementa uma estimativa de erro de temporização *dentro* de um
        #   único símbolo. Ambas são abordagens válidas para rastrear imperfeições
        #   dinâmicas usando os pilotos.
        #
        slope = 0  # SUBSTITUA PELO SEU CÓDIGO

        # Calcula a média da inclinação de fase (slope) ao longo dos últimos `L` símbolos
        # OFDM. Isso suaviza a estimativa, tornando-a mais robusta ao ruído.
        # Um erro de temporização muda muito lentamente, então fazer a média de várias
        # estimativas consecutivas nos dá um resultado mais estável.
        average_slope_filter[1:] = average_slope_filter[:-1]
        average_slope_filter[0] = slope
        average_slope = np.sum(average_slope_filter) / min(i+1, L)

        # Correção de fase residual entre portadoras
        k = np.fft.fftshift(np.arange(-32, 32))
        applied_correction = k * average_slope

        # Aplica a correção de fase linear (derivada da inclinação ou 'slope') a cada
        # subportadora do símbolo. Este símbolo (`corrected_symbol1`) já teve o erro de
        # fase *comum* (`theta`) removido. Agora, estamos corrigindo o erro de fase
        # que varia de uma subportadora para outra, que é a manifestação de um
        # erro de temporização.
        #
        # - Detalhe da Operação:
        #   - `applied_correction`: É um vetor que contém a correção de fase angular
        #     necessária para *cada* subportadora. Ele representa uma rampa de fase
        #     linear que anula o efeito do erro de temporização.
        #   - A multiplicação elemento a elemento por `np.exp(-1j * applied_correction)`
        #     derrota cada subportadora pelo seu ângulo de correção específico.
        #
        # - Referência ao Livro-texto: Esta é a aplicação direta da propriedade de
        #   deslocamento no tempo da Transformada de Fourier, conforme discutido
        #   na Seção 7.3.6 (Página 535) e ilustrado na Figura 7-51.
        corrected_symbol2 = corrected_symbol1 * np.exp(-1j * applied_correction)

        # Atualiza os coeficientes do equalizador para incorporar a correção de deriva
        # de temporização. Em vez de apenas corrigir o símbolo atual, esta linha
        # "ensina" o equalizador sobre o erro de temporização que ele acabou de medir,
        # para que a correção seja aplicada de forma mais suave e contínua nos
        # próximos símbolos.
        #
        # - Detalhe da Operação:
        #   - `applied_correction / L`: Esta é a parte mais importante. Estamos
        #     aplicando apenas uma pequena fração (1/L) da correção medida aos
        #     coeficientes do equalizador. Isso atua como um filtro passa-baixa,
        #     suavizando a atualização e evitando que o equalizador reaja
        #     bruscamente a medições de ruído em um único símbolo. Ele permite que o
        #     equalizador *rastreie* (track) a deriva de temporização lentamente.
        #
        # - Referência ao Livro-texto: Este é um conceito de **equalização adaptativa**
        #   na prática, relacionado aos princípios do algoritmo LMS (Seção 4.4, p. 286),
        #   onde os coeficientes de um filtro são atualizados iterativamente com base
        #   em um sinal de erro.
        #
        # - Referência gr-ieee802-11: No arquivo `lib/frame_equalizer_impl.cc`, o
        #   rastreamento de temporização (`d_er`) é usado para ajustar a fase por
        #   subportadora em `current_symbol[i] *= exp(gr_complex(0, 2 * M_PI * ...))`.
        #   A abordagem em Python é conceitualmente semelhante, mas opta por
        #   integrar lentamente essa correção nos próprios coeficientes do equalizador,
        #   em vez de aplicá-la como um passo separado a cada vez.
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
    Desembaralha um fluxo de bits que foi previamente embaralhado
    pelo embaralhador padrão IEEE 802.11a/g.

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    Sua tarefa é reverter o processo da função `scramble`. A beleza da criptografia
    de fluxo aditiva (como este embaralhador) é que o mesmo processo de
    embaralhamento, quando aplicado aos dados embaralhados, recupera os dados
    originais. Isso ocorre porque o XOR é sua própria operação inversa
    (ou seja, `(A XOR B) XOR B = A`).

    Portanto, a implementação desta função será quase idêntica à da função `scramble`,
    ou pode até fazer uso da própria função `scramble` após determinar o estado inicial.

    ## Referências Cruzadas:

    - **Norma IEEE 802.11a:** Seção 17.3.5.4 ("PLCP DATA scrambler and descrambler")
      Ela especifica o polinômio gerador `S(x) = x^7 + x^4 + 1`, que corresponde
      ao `feedback = state[6] ^ state[3]`. Ela também contém a "chave" para
      a decodificação: "The seven LSBs of the SERVICE field will be set to
      all zeros prior to scrambling to enable estimation of the initial state of
      the scrambler in the receiver."

    - **gr-ieee802-11:** No arquivo `lib/decode_mac.cc`, a função `decode_mac::descramble`
      implementa exatamente esta lógica. Ela primeiro extrai o estado inicial
      dos primeiros 7 bits recebidos e depois usa esse estado para desembaralhar
      o resto do pacote.

    - **Conceito Teórico Externo:** Este tipo de embaralhador é tecnicamente conhecido
      como um Gerador de Sequência Pseudoaleatória (PRNG) implementado com um
      Registrador de Deslocamento com Retroalimentação Linear (Linear Feedback Shift
      Register - LFSR), usado em uma configuração de cifra de fluxo síncrona.
    """
    return np.array([], dtype=np.uint8)

def convolutional_decoder(soft_bits):
    """
    Decodifica um fluxo de "soft bits" (LLRs) usando o algoritmo de Viterbi.

    Esta função implementa o processo inverso da codificação convolucional,
    encontrando a sequência de bits de informação mais provável que, ao ser
    codificada, teria gerado o fluxo de bits recebido (corrompido por ruído).

    TAREFA DO ALUNO: Implementar a decodificação Viterbi.

    --------------------------------------------------------------------------
    ## Lógica de Implementação:

    1.  **Entrada "Soft":** A função receberá `soft_bits`. Estes não são 0s e 1s
        ("hard bits"), mas sim valores de ponto flutuante (Log-Likelihood Ratios - LLRs).
        Um valor positivo grande indica alta confiança de que o bit era '1', um valor
        negativo grande indica alta confiança de que era '0', e valores próximos de
        zero indicam incerteza. Usar "soft bits" em vez de "hard bits" melhora
        significativamente o desempenho da decodificação (cerca de 2 dB de ganho).

    2.  **Saída:** A função `convolutional_decoder` deve retornar um array NumPy
        com os bits de informação decodificados (0s e 1s).

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Livro-texto:**
        - **Seção 5.6.3, "Binary Convolutional Coding" (Página 369):** Descreve
          o processo de codificação e a estrutura do codificador.
        - **Seção 5.6.4, "Convolutional Decoder (The Viterbi Algorithm)" (Página 373):**
          Explica a teoria por trás do algoritmo de Viterbi. A **Figura 5-79 (Página 372)**
          mostra o diagrama de treliça, que é a estrutura de dados fundamental que o
          algoritmo percorre. A **Figura 5-82 (Página 375)** ilustra os conceitos de
          métrica de caminho (path metric) e métrica de ramo (branch metric), que são
          os "custos" que o algoritmo minimiza para encontrar o melhor caminho.

    - **Norma IEEE 802.11a:**
        - **Seção 17.3.5.5, "Convolutional encoder":** Especifica os polinômios
          geradores `g0 = 133(oct)` e `g1 = 171(oct)` e o comprimento da restrição `K=7`,
          que são os parâmetros usados para construir a treliça.

    - **gr-ieee802-11:**
        - O arquivo **`lib/viterbi_decoder/base.h`** e suas implementações
          (`viterbi_decoder_x86.cc`, `viterbi_decoder_generic.cc`) contêm uma
          implementação C++ altamente otimizada do decodificador Viterbi.

    - **Conceito Teórico Externo:** O algoritmo de Viterbi é um exemplo clássico de
      **programação dinâmica**, onde um problema complexo é resolvido encontrando-se
      soluções ótimas para subproblemas menores.
    """
    return np.array([], dtype=int)

def create_deinterleaving_pattern(n_cbps, n_bpsc):
    """
    Cria e armazena em cache o padrão de permutação para o de-interleaving de um
    símbolo OFDM, conforme a norma IEEE 802.11a.

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    O interleaving, no transmissor, embaralha a ordem dos bits codificados antes
    do mapeamento para as subportadoras. O objetivo é espalhar bits que eram
    consecutivos no tempo. Se uma parte do espectro (um grupo de subportadoras
    adjacentes) for atenuada por fading seletivo, os erros de bit resultantes
    estarão espalhados após o de-interleaving no receptor, em vez de ocorrerem
    em uma rajada. Isso aumenta drasticamente a eficácia do decodificador
    Viterbi, que lida mal com rajadas de erros.

    Sua tarefa é gerar o padrão de *de-interleaving*, que é a permutação *inversa*
    daquela realizada no transmissor.

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Norma IEEE 802.11a:** A Seção 17.3.5.6 ("Data interleaving", Página 25)
      contém as equações exatas (Equações 15 e 16) para as permutações.

    - **gr-ieee802-11:** O arquivo `lib/utils.cc` contém a função `interleave`, que
      implementa a permutação direta em C++. O `decode_mac.cc` aplica o
      de-interleaving. O estudo dessas funções pode ajudar a entender o fluxo.
    """
    return np.array([], dtype=int)

def demapper_ofdm(symbols_iq, n_bpsc):
    """
    Realiza o de-mapeamento "soft" de símbolos de constelação complexos para
    Log-Likelihood Ratios (LLRs), também conhecidos como "soft bits".

    TAREFA DO ALUNO: Implementar esta função.

    --------------------------------------------------------------------------
    ## Objetivo:
    O de-mapeador converte os símbolos complexos recebidos após a equalização
    (que ainda são valores de ponto flutuante contendo ruído) em uma estimativa
    da probabilidade de cada bit codificado ser um '0' ou um '1'. O LLR é uma
    métrica logarítmica dessa probabilidade.

    Para constelações QAM mais simples como BPSK e QPSK (que são usadas em 802.11a/g),
    uma aproximação excelente do LLR para um bit é simplesmente o valor da projeção
    do símbolo no eixo correspondente (I ou Q). Um valor real positivo grande
    indica alta probabilidade de o bit ser '1' (no mapeamento do livro), enquanto
    um valor negativo grande indica alta probabilidade de ser '0'.

    Sua tarefa é implementar este de-mapeamento para BPSK e QPSK.

    --------------------------------------------------------------------------
    ## Lógica de Implementação:

    A função deve ter um comportamento diferente dependendo do número de bits por
    subportadora (`n_bpsc`).

    **1. Caso BPSK (`n_bpsc == 1`):**
        - No BPSK, toda a informação está no eixo real (I).
        - O LLR para o único bit é simplesmente o valor da componente real do
          símbolo complexo.
        - Você deve extrair a parte real de cada símbolo em `symbols_iq` e
          retornar o array resultante.

    **2. Caso QPSK (`n_bpsc == 2`):**
        - No QPSK, um bit é mapeado no eixo real (I) e o outro no eixo imaginário (Q).
        - O LLR para o primeiro bit de cada símbolo é a sua componente real.
        - O LLR para o segundo bit de cada símbolo é a sua componente imaginária.
        - Você deve:
            a. Extrair um array com todas as componentes reais (`soft_bits_i`).
            b. Extrair um array com todas as componentes imaginárias (`soft_bits_q`).
            c. Intercalar (interleave) esses dois arrays para produzir um único fluxo
               de LLRs, na ordem correta: `[i_0, q_0, i_1, q_1, i_2, q_2, ...]`.

    **3. Outros Casos:**
        - Para este exercício, se `n_bpsc` for algo diferente de 1 ou 2, você
          pode lançar um `ValueError`, pois o de-mapeamento para 16-QAM e 64-QAM
          requer cálculos de LLR mais complexos que não são abordados aqui.

    --------------------------------------------------------------------------
    ## Referências Cruzadas:

    - **Livro-texto:** Seção 5.6.2, "Forward Error Correction and the Log Likelihood
      Scaling Process" (Página 357). O texto explica a diferença entre
      "hard bits" e "soft bits" e a importância de usar a informação de
      confiança (magnitude do valor soft) no processo de decodificação.

    - **Norma IEEE 802.11a:** A Figura 116 (página 28) mostra o mapeamento de
      bits para as constelações, que é o processo que esta função inverte.

    - **gr-ieee802-11:** Este processo é implementado no arquivo
      `lib/constellations_impl.cc`.
    """
    return np.array([], dtype=float)
