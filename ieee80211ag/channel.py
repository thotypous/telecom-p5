# =============================================================================
# Modelo para Simulação de Canal
# =============================================================================

import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hann
from scipy.interpolate import CubicSpline
import logging

DEFECT_MODE = {
    'Multipath': 1,
    'ThermalNoise': 1,
    'PhaseNoise': 1,
    'Freq_Offset': 1,
    'IQ_Imbalance': 1,
    'TimingOffset': 1,
    'TimingDrift': 1,
}

DEFECT_SETTINGS = {
    'SampleRate': 40e6,
    'NumberOfTaps': 40,
    'DelaySpread': 150e-9,
    'SNR_dB': 35,
    'PhaseNoiseProfile': np.array([
        [1e3, 10e3, 20e3, 35e3, 50e3, 80e3, 90e3, 100e3, 120e3, 150e3, 200e3, 300e3, 500e3, 1e6],
        [-70, -72, -72, -74, -76, -85, -90, -95, -100, -105, -110, -120, -130, -140]
    ]),
    'FrequencyOffset': -100e3,
    'PhaseImbalance': np.pi / 2000,
    'AmplitudeImbalance_dB': -0.1,
    'Sample_Offset': -1,
    'Drift_ppm': -80,
}

def default_defect_model(tx_samples):
    return defect_model(tx_samples, DEFECT_SETTINGS, DEFECT_MODE)

def defect_model(tx_samples, settings, mode):
    """
    Aplica uma cascata de defeitos de canal a um sinal transmitido.
    Esta função modela as imperfeições do mundo real de um canal de comunicação sem fio.
    Referência: Livro-texto, Capítulo 6, Figura 6-16.
    """
    # 3. Gerando Filtro de Múltiplos Percursos
    fir_taps = get_multipath_filter(settings['SampleRate'], settings['DelaySpread'], settings['NumberOfTaps'])
    if mode['Multipath'] == 1:
        tx_samples = lfilter(fir_taps, [1.0], tx_samples)
        var_output = np.var(tx_samples)
        if var_output > 0:
             tx_samples /= np.sqrt(var_output)

    # 4. Gerando Ruído Branco Gaussiano Aditivo (Ruído Térmico)
    if mode['ThermalNoise'] == 1:
        tx_samples += generate_awgn(tx_samples, settings['SNR_dB'])

    # 5. Adicionando Ruído de Fase
    if mode['PhaseNoise'] == 1:
        number_of_samples = len(tx_samples)
        ph_noise, rmsn = phase_noise_generator(settings['SampleRate'], number_of_samples,
                                           settings['PhaseNoiseProfile'][1, :],
                                           settings['PhaseNoiseProfile'][0, :])
        ph_noise -= np.mean(ph_noise)
        phase_noise_signal = np.exp(1j * ph_noise)
        logging.info(f"Integrated Phase Noise1: {rmsn * 57.3}")
        tx_samples *= phase_noise_signal

    # 6. Adicionando Deslocamento de Frequência
    if mode['Freq_Offset'] == 1:
        offset_signal = np.exp(1j * 2 * np.pi * np.arange(1, len(tx_samples) + 1) * settings['FrequencyOffset'] / settings['SampleRate'])
        tx_samples *= offset_signal

    # 7. Adicionando Desequilíbrio de I/Q
    if mode['IQ_Imbalance'] == 1:
        i_gain = 10**(0.5 * settings['AmplitudeImbalance_dB'] / 20)
        q_gain = 10**(-0.5 * settings['AmplitudeImbalance_dB'] / 20)
        tx_samples = iq_imbalance(tx_samples, settings['PhaseImbalance'], i_gain, q_gain)

    # 8. Adicionar Deslocamento de Temporização
    if mode['TimingOffset'] == 1 and len(tx_samples) > abs(settings['Sample_Offset']) + 2:
        tx_samples = cause_timing_offset(tx_samples, settings['Sample_Offset'])

    # 9. Adicionar Deriva de Temporização
    if mode['TimingDrift'] == 1 and len(tx_samples) > 1:
        tx_samples = cause_timing_drift(tx_samples, settings['Drift_ppm'])

    output_waveform = tx_samples
    return output_waveform, fir_taps

def cause_timing_drift(input_seq, drift):
    """
    Simula a deriva de temporização (timing drift) em um sinal.
    A deriva de temporização ocorre devido a pequenas diferenças entre as frequências
    dos osciladores de referência do transmissor e do receptor (e.g., TCXO).
    Isso causa uma compressão ou expansão gradual da forma de onda no tempo.
    A função interpola o sinal de entrada em novos pontos de tempo para simular este efeito.
    Referência: Livro-texto, Seção 6.3.7, "Timing Drift".
    Referência Norma IEEE 802.11a: A tolerância do relógio de símbolo é de +/- 20 ppm (Seção 17.3.9.5).
    """
    # Calcula o novo passo de amostragem. Um drift positivo significa que o relógio
    # do transmissor é mais rápido que o do receptor, então "pulamos" amostras.
    sample_step = 1 + drift / 1e6
    input_index = np.arange(1, len(input_seq) + 1)
    # Gera os novos instantes de tempo onde o sinal será amostrado
    output_index = np.arange(1, len(input_seq) + 1, sample_step)

    # A interpolação requer que os pontos de consulta estejam dentro do intervalo dos dados.
    # Garante que os pontos de interpolação estejam dentro dos limites dos dados de entrada.
    output_index = output_index[output_index <= len(input_seq)]
    output_index = output_index[(output_index >= input_index[0]) & (output_index <= input_index[-1])]

    # Usa a interpolação por spline cúbico para estimar os valores nos novos pontos.
    cs = CubicSpline(input_index, input_seq)
    output = cs(output_index)
    return output

def cause_timing_offset(input_seq, sample_delay):
    """
    Simula um deslocamento de temporização (timing offset) estático em um sinal.
    Este deslocamento representa um erro fixo no instante de amostragem no receptor.
    A função usa interpolação para reamostrar o sinal com o deslocamento desejado.
    Referência: Livro-texto, Seção 6.3.7, "Timing Offset" e Figura 6-50.
    """
    sample_step = 1
    input_index = np.arange(1, len(input_seq) + 1)
    output_index = np.arange(1 + sample_delay, len(input_seq) + 1, sample_step)

    # A interpolação requer que os pontos de consulta estejam dentro do intervalo dos dados.
    # Garante que os pontos de interpolação estejam dentro dos limites dos dados de entrada.
    output_index = output_index[(output_index >= input_index[0]) & (output_index <= input_index[-1])]

    cs = CubicSpline(input_index, input_seq)
    output = cs(output_index)
    return output

def generate_awgn(input_seq, snr):
    """
    Gera ruído branco aditivo Gaussiano (AWGN) com uma dada Relação Sinal-Ruído (SNR).
    O ruído térmico em sistemas de comunicação é bem modelado como AWGN.
    A potência do ruído é calculada com base na potência do sinal de entrada e na SNR desejada.
    Referência: Livro-texto, Seção 6.3.3, "Gaussian White Noise in Receivers".
    Referência gr-ieee802-11: Semelhante à funcionalidade do bloco `channels_channel_model`.
    """
    # Calcula a potência média do sinal de entrada (variância para sinais de média zero).
    mean_square = np.mean(np.abs(input_seq)**2)
    # Calcula a potência de ruído necessária para atingir a SNR alvo.
    noise_power = mean_square / (10**(snr / 10))
    std_noise = np.sqrt(noise_power)

    # Gera ruído complexo com componentes real e imaginário independentes e
    # com distribuição Gaussiana. O fator 0.70711 (1/sqrt(2)) normaliza a potência.
    noise = std_noise * (0.70711 * np.random.randn(len(input_seq)) +
                         1j * 0.70711 * np.random.randn(len(input_seq)))
    return noise

def get_multipath_filter(sample_rate, delay_spread, n):
    """
    Gera coeficientes de um filtro FIR para simular um canal de múltiplos percursos.
    O modelo assume um perfil de potência exponencialmente decrescente, que é
    típico para canais de rádio em ambientes internos (modelo de Naftali).
    Referência: Livro-texto, Seção 6.3.1, "Multipath Distortion (No Mobility)" e Figura 6-20.
    """
    ts = 1 / sample_rate
    trms = delay_spread

    # Calcula a variância para cada coeficiente do filtro (tap).
    n_range = np.arange(n)
    exp_variance = np.exp(-n_range * ts / trms)

    # Gera os coeficientes do filtro FIR como variáveis aleatórias complexas Gaussianas
    # com a variância calculada.
    fir_taps = np.zeros(n, dtype=complex)
    for i in range(n):
        fir_taps[i] = (np.sqrt(exp_variance[i]) * np.random.randn() +
                       1j * np.sqrt(exp_variance[i]) * np.random.randn())
    return fir_taps

def iq_imbalance(tx_samples, phase_imbalance, i_gain, q_gain):
    """
    Simula desequilíbrio de I/Q (amplitude e fase) no sinal.
    Desequilíbrio de fase ocorre quando os osciladores locais de I e Q não estão
    exatamente 90 graus defasados. Desequilíbrio de amplitude ocorre quando os
    ganhos dos caminhos de I e Q são diferentes.
    Referência: Livro-texto, Seção 6.3.6, "Imbalances in IQ Modulators" e Figura 6-45.

    NOTA DE TRADUÇÃO: O arquivo IQImbalance.m original tinha uma assinatura
    de função e implementação que não correspondiam à sua chamada
    em Defect_Model.m. O código MATLAB original falharia na execução.
    Esta função foi reescrita para implementar um modelo de desequilíbrio de I/Q
    padrão que corresponde aos parâmetros passados de Defect_Model.m
    (tx_samples, phase_imbalance, i_gain, q_gain).
    O modelo implementado é:
    s_out(t) = g_i * i(t) + j * g_q * ( q(t) * cos(phi) + i(t) * sin(phi) )
    """
    i_in = np.real(tx_samples)
    q_in = np.imag(tx_samples)

    # Simula o erro de fase (cross-talk do componente I para o Q).
    q_phase_imbalanced = q_in * np.cos(phase_imbalance) + i_in * np.sin(phase_imbalance)

    # Aplica os ganhos de amplitude.
    i_out = i_gain * i_in
    q_out = q_gain * q_phase_imbalanced

    return i_out + 1j * q_out

def phase_noise_generator(sample_rate, number_of_samples, dbc, freq):
    """
    Gera uma forma de onda de ruído de fase no domínio do tempo a partir de um perfil de
    densidade espectral de potência (PSD) no domínio da frequência.

    O processo simula como um oscilador real não produz uma única frequência pura, mas sim
    um espectro de frequências centrado na portadora, onde a potência decai à medida que
    nos afastamos dela. Esta função traduz esse perfil de frequência em uma
    forma de onda de desvio de fase temporal, `θ(t)`, que pode ser aplicada a um
    sinal ideal para simular a imperfeição.

    O método segue uma abordagem clássica de modelagem de ruído colorido:
    1. Interpola o perfil de ruído de fase (dado em pontos esparsos) para uma grade fina.
    2. Calcula o erro de fase RMS (Root Mean Square) total, que representa a potência
       total do ruído de fase. Este será o nosso alvo de normalização.
    3. Projeta um filtro FIR cuja resposta em frequência corresponde à raiz quadrada
       do perfil de potência do ruído (o perfil de magnitude).
    4. Gera ruído branco gaussiano (que tem um espectro plano).
    5. Passa o ruído branco pelo filtro FIR. A saída será um "ruído colorido" com o
       espectro de potência desejado.
    6. Faz o upsample do sinal de ruído para a taxa de amostragem final e o normaliza
       para que seu valor RMS corresponda ao `rms_pe` calculado na etapa 2.

    - Referência ao Livro-texto: Seção 6.3.4, "Phase Noise" (Página 453) e
      a Figura 6-38 ("The Creation of Phase Noise", Página 457), que mostra
      o diagrama de blocos exato deste processo.
    """
    # --- Etapa 1: Interpolar o Perfil de Ruído de Fase para uma Grade Linear Fina ---
    # Perfis de ruído de fase são geralmente definidos em poucos pontos em uma escala logarítmica.
    # Para o processamento digital (integração, IFFT), precisamos de uma representação
    # em uma grade de frequência linear e densa.

    # Adiciona pontos de limite ao perfil para garantir um comportamento estável da interpolação
    # nas bordas. -200 dBc é um valor muito baixo que efetivamente define o "chão" do ruído.
    max_freq = freq[-1]
    freq_index = np.concatenate(([0], freq, [max_freq + 1, 2.1e6]))
    dbc_power = np.concatenate(([-200], dbc, [-200, -200]))

    # Interpola o perfil para uma grade linear de 1 kHz de resolução.
    new_frequencies = np.arange(0, 2e6 + 1e3, 1e3)
    new_dbc_power = np.interp(new_frequencies, freq_index, dbc_power)
    # Converte de dBc/Hz para potência linear para cálculos matemáticos.
    new_linear_power = 10**(new_dbc_power / 10)

    # --- Etapa 2: Calcular o Erro de Fase RMS Total ---
    # Integra numericamente a densidade espectral de potência (PSD) de banda lateral única (SSB)
    # para encontrar a potência total do ruído de fase em um lado da portadora.
    ssb_power = np.trapezoid(new_linear_power, new_frequencies)
    # A potência total do ruído é o dobro da potência SSB (considerando ambas as bandas laterais).
    # O erro de fase RMS (em radianos) é a raiz quadrada da potência total. Este é o nosso alvo.
    # Referência Livro-texto: Fórmula para RmsPhaseError na página 456.
    rms_pe = np.sqrt(2 * ssb_power)

    # --- Etapa 3: Projetar o Filtro FIR usando o Método de Amostragem de Frequência ---
    # O objetivo é criar um filtro cuja resposta em magnitude se assemelhe à raiz quadrada da PSD do ruído.
    n_fft = 2000  # Tamanho da IFFT para gerar os coeficientes do filtro
    # Define as frequências para a IFFT, baseadas em uma taxa de amostragem interna de 4MHz.
    freq_pos = np.arange(n_fft // 2) * 4e6 / n_fft
    freq_neg = np.arange(-n_fft // 2, 0) * 4e6 / n_fft

    # Interpola o perfil de potência linear para as frequências da IFFT.
    pow_pos = np.interp(freq_pos, new_frequencies, new_linear_power, left=1e-20, right=1e-20)
    pow_neg = np.interp(np.abs(freq_neg), new_frequencies, new_linear_power, left=1e-20, right=1e-20)

    linear_power = np.concatenate((pow_pos, pow_neg))
    # A resposta em magnitude do filtro é a raiz quadrada da resposta em potência.
    magnitude = np.sqrt(linear_power)
    # A IFFT da resposta em frequência nos dá a resposta ao impulso (coeficientes do filtro).
    temp = np.fft.ifft(magnitude)

    # Reordena os coeficientes da IFFT para o formato padrão de filtro FIR (simétrico em torno do centro).
    fir_taps = np.concatenate((temp[n_fft//2:], temp[:n_fft//2]))
    # Aplica uma janela de Hanning para suavizar a resposta ao impulso e reduzir artefatos (ringing).
    # Referência Livro-texto: Seção 2.3.3, "The Effect of Windows" (Página 163).
    h = hann(n_fft + 2, sym=False)
    h1 = h[1:n_fft+1]
    fir_taps *= h1

    # --- Etapa 4: Filtrar Ruído Branco para Gerar Ruído Colorido ---
    # Define a taxa de amostragem interna e gera ruído branco gaussiano (WGN).
    sample_ratio = sample_rate / 4e6
    samples_at_4mhz = int(np.ceil(number_of_samples / sample_ratio))
    wgn = np.random.randn(n_fft + samples_at_4mhz)

    # Passa o WGN (espectro plano) pelo filtro FIR. A saída terá um espectro com o formato
    # da resposta em frequência do filtro, criando o "ruído colorido" desejado.
    filter_output = lfilter(fir_taps, [1], wgn)
    # Descarta o transiente inicial do filtro.
    out_4mhz = filter_output[n_fft//2 -1 : n_fft//2 -1 + samples_at_4mhz]

    # --- Etapa 5: Upsample para a Taxa de Amostragem Desejada e Normalização Final ---
    # O ruído foi gerado a 4 MS/s. Agora, ele é interpolado (upsampled) para a
    # taxa de amostragem final (`sample_rate`) do sistema.
    sample_time_4mhz = np.arange(1, len(out_4mhz) + 1)
    new_sample_time = np.arange(1, number_of_samples + 1) / sample_ratio

    # Garante que os pontos de consulta da interpolação estejam dentro do intervalo de dados válido.
    new_sample_time = new_sample_time[new_sample_time <= sample_time_4mhz[-1]]
    new_sample_time = new_sample_time[new_sample_time >= sample_time_4mhz[0]]

    cs = CubicSpline(sample_time_4mhz, out_4mhz)
    output_interp = cs(new_sample_time)

    # Preenche o restante do buffer, se a interpolação não gerar amostras suficientes.
    if len(output_interp) < number_of_samples:
        padding = np.zeros(number_of_samples - len(output_interp))
        output = np.concatenate((output_interp, padding))
    else:
        output = output_interp[:number_of_samples]

    # Normaliza a potência (variância) da forma de onda de ruído gerada para
    # corresponder exatamente ao erro de fase RMS (`rms_pe`) calculado a partir
    # da especificação original. Isso garante que o ruído tenha a intensidade correta.
    out_temp = output[:number_of_samples]
    rms_output = np.sqrt(np.var(out_temp))

    # Evita divisão por zero se a saída for nula.
    if rms_output == 0:
        rms_output = 1e-9

    out = out_temp * rms_pe / rms_output

    return out, rms_pe
