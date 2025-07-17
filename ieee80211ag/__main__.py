"""
Tradução para Python do Code_Chapter7_Section4_Final_OFDM_Modem_TestBench do MATLAB

Este script simula um modem OFDM completo, similar às normas 802.11a/g,
incluindo transmissor, um modelo de canal com vários defeitos e um receptor.
O desempenho é avaliado usando a Magnitude do Vetor de Erro (EVM).

Além da opção de usar um modelo de canal, este script acrescenta a opção de
receber quadros a partir de um sinal I/Q gravado com um SDR.

A tradução visa ser fiel ao código MATLAB original, gerando resultados
consistentes.

NOTA: Este código foi modificado para usar tons piloto pseudoaleatórios
conforme a norma IEEE 802.11a, em vez dos pilotos fixos da simulação original.
Além disso, foram adicionadas novas etapas, que não estavam incluídas no
código que acompanha o livro, a fim de implementar um transceptor completo.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

from .channel import *
from .tx import *
from .rx import *

# =============================================================================
# Script Principal (TestBench ou Receptor)
# =============================================================================
def main():
    """
    Referência: Livro-texto, Seção 7.4, "802.11a Performance Evaluation and Transceiver Code".
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--testbench", action="store_true", help="Executar um teste com dados sintéticos")
    group.add_argument("--iq", type=str, help="Ler dados de uma gravação I/Q de um SDR")
    group.add_argument("--npz", type=str, help="Ler dados de um arquivo npz com uma gravação cortada em trechos")
    parser.add_argument("--correct_frequency_offset", type=bool, default=True)
    parser.add_argument("--use_max_ratio_combining", type=bool, default=True)
    parser.add_argument("--sample_advance", type=int, default=1)
    parser.add_argument("--random_seed", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Configuração Geral da Simulação
    psdu_length = 1000
    rate_key = 0b0101  # QPSK 1/2
    transmitter_choice = 1  # 0/1 = upsampled / IFFT128

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    if args.testbench:
        # 2. Configuração do Transmissor
        logging.info("Iniciando Transmissor...")
        # Gera os bytes do PSDU exceto pelo CRC (4 bytes), que será incluído pela função ofdm_transmitter
        mac_frame_bytes = np.random.randint(0, 256, psdu_length - 4, dtype=np.uint8)
        sample_output, tx_symbol_stream = ofdm_transmitter(mac_frame_bytes, rate_key, transmitter_choice)
        tx_output_clean = np.concatenate((np.zeros(10), sample_output, np.zeros(20)))
        logging.info("Transmissão concluída.")

        # 3. Modelo de Defeitos
        logging.info("Aplicando defeitos do canal...")
        tx_output, fir_taps = default_defect_model(tx_output_clean)
        logging.info("Defeitos aplicados.")

        rx_waveform_20mhz = tx_output[::2]
        rx_waveforms = [rx_waveform_20mhz]
    elif args.iq is not None:
        arr = np.fromfile(args.iq, dtype=np.complex64)
        rx_waveforms = encontrar_trechos(arr, limiar=.004, max_separacao=100, padding=50)
        #np.savez_compressed('record.npz', *rx_waveforms[:100])
    elif args.npz is not None:
        rx_waveforms = [arr for arr in np.load(args.npz).values()]

    for rx_waveform_20mhz in rx_waveforms:
        # 4. Código do Receptor
        logging.info("Iniciando Receptor...")
        corrected_symbols = ofdm_receiver(rx_waveform_20mhz,
            sample_advance=args.sample_advance,
            correct_frequency_offset=args.correct_frequency_offset,
            number_of_ofdm_symbols=1000,
            use_max_ratio_combining=args.use_max_ratio_combining)
        logging.info("Recepção concluída.")

        if corrected_symbols.size == 0:
            logging.warning("A recepção falhou. Pulando quadro.")
            continue

        # Decodifica quadro
        decoded_params = decode_signal_field(corrected_symbols[:48])
        print("\n--- Decodificação do Campo SIGNAL ---")
        print(f"Taxa de transmissão: {decoded_params['rate_info']['name']} ({decoded_params['rate_info']['Mbps']} Mbps)")
        print(f"Comprimento do PSDU: {decoded_params['length']} bytes")
        print(f"Verificação de Paridade: {'OK' if decoded_params['parity_ok'] else 'FALHOU'}")
        print(f"Verificação da Cauda: {'OK' if decoded_params['tail_ok'] else 'FALHOU'}")
        print()

        print("--- Decodificação dos Dados ---")
        received_mac_frame_bytes, tail_ok, crc_ok = decode_data_symbols(corrected_symbols[48:], decoded_params['rate_info'], decoded_params['length'])
        if args.testbench:
            print(f"Bytes iguais aos originais: {np.sum(received_mac_frame_bytes == mac_frame_bytes)/len(mac_frame_bytes):.1%}")
        else:
            print(bytes(received_mac_frame_bytes))
        print(f"Verificação da Cauda: {'OK' if tail_ok else 'FALHOU'}")
        print(f"Verificação de CRC: {'OK' if crc_ok else 'FALHOU'}")
        print()

        if args.testbench:
            # 5. Avaliação de Desempenho
            logging.info("Avaliando desempenho...")
            error_vectors = tx_symbol_stream - corrected_symbols
            average_error_vector_power = np.mean(np.abs(error_vectors)**2)

            # Evita log de zero
            if average_error_vector_power == 0:
                average_error_vector_power = 1e-12

            evm = 10 * np.log10(average_error_vector_power / 1)
            print(f"EVM = {evm:.4f} dB")

            # Cálculo do EVM vs. Tempo
            num_processed_symbols = len(corrected_symbols) // 48
            error_time = np.zeros(num_processed_symbols)
            for i in range(num_processed_symbols):
                start = i * 48
                stop = start + 48
                current_symbol = corrected_symbols[start:stop]
                ideal_symbol = tx_symbol_stream[start:stop]
                error_vector = ideal_symbol - current_symbol
                error_time[i] = np.mean(np.abs(error_vector)**2)

            # Cálculo do EVM vs. Frequência
            error_frequency = np.zeros(48)
            for i in range(num_processed_symbols):
                start = i * 48
                stop = start + 48
                current_symbol = corrected_symbols[start:stop]
                ideal_symbol = tx_symbol_stream[start:stop]
                error_vector = ideal_symbol - current_symbol
                error_frequency += (np.abs(error_vector)**2) / num_processed_symbols

            error_frequency[error_frequency == 0] = 1e-12
            evm_frequency = 10 * np.log10(error_frequency)

            error_time[error_time == 0] = 1e-12
            evm_time = 10 * np.log10(error_time)

            # Índices para o gráfico de frequência
            pos_index = np.concatenate((np.arange(1, 7), np.arange(8, 21), np.arange(22, 27)))
            neg_index = np.concatenate((np.arange(38, 43), np.arange(44, 57), np.arange(58, 64))) - 64

            # Resposta de frequência do filtro FIR
            f = np.arange(-0.5, 0.501, 0.001)
            n = np.arange(len(fir_taps))
            response = np.zeros(len(f), dtype=complex)
            for d_idx, d_val in enumerate(f):
                E = np.exp(1j * 2 * np.pi * n * d_val)
                response[d_idx] = np.dot(fir_taps, E.conj().T)

            mag_response = 20 * np.log10(np.abs(response))
            mag_response_norm = mag_response - np.max(mag_response)

            # Plotagem dos resultados
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Análise de Desempenho do Modem OFDM', fontsize=16)

            # EVM vs Frequência
            axs[0, 0].plot(pos_index, evm_frequency[:24], 'k.')
            axs[0, 0].plot(neg_index, evm_frequency[24:], 'k.')
            axs[0, 0].set_title('EVM vs. Frequência')
            axs[0, 0].set_xlabel('Tons (Subportadoras)')
            axs[0, 0].set_ylabel('dB')
            axs[0, 0].set_xlim(-27, 27)
            axs[0, 0].set_ylim(-40, 5)
            axs[0, 0].grid(True)

            # EVM vs Tempo
            axs[0, 1].plot(evm_time, 'k')
            axs[0, 1].set_title('EVM vs. Tempo')
            axs[0, 1].set_xlabel('Símbolos')
            axs[0, 1].set_ylabel('dB')
            if len(evm_time) > 1:
              axs[0, 1].set_xlim(0, len(evm_time) -1)
            axs[0, 1].set_ylim(-40, -10)
            axs[0, 1].grid(True)

            # Resposta de Magnitude do Filtro de Múltiplos Percursos
            axs[1, 0].plot(f, mag_response_norm, 'k')
            axs[1, 0].set_title('Resposta de Magnitude do Filtro de Múltiplos Percursos')
            axs[1, 0].set_xlabel('Frequência Normalizada')
            axs[1, 0].set_ylabel('dB')
            axs[1, 0].set_xlim(-13/64, 13/64)
            axs[1, 0].set_ylim(-25, 5)
            axs[1, 0].grid(True)

            # Coeficientes FIR
            axs[1, 1].stem(np.abs(fir_taps), linefmt='k-', markerfmt='k.', basefmt='k-')
            axs[1, 1].set_title('Coeficientes do Filtro FIR (Magnitude)')
            axs[1, 1].set_xlabel('Amostras')
            axs[1, 1].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Gráfico da Constelação
        plt.figure(figsize=(8, 8))
        plt.plot(np.real(corrected_symbols[:48]), np.imag(corrected_symbols[:48]), 'k.', markersize=8)
        plt.title('Constelação Recebida Após Equalização (SIGNAL symbol)')
        plt.xlabel('Real')
        plt.ylabel('Imaginário')
        plt.grid(True)
        plt.axis('square')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.figure(figsize=(8, 8))
        plt.plot(np.real(corrected_symbols[48:]), np.imag(corrected_symbols[48:]), 'k.', markersize=8)
        plt.title('Constelação Recebida Após Equalização (DATA symbols)')
        plt.xlabel('Real')
        plt.ylabel('Imaginário')
        plt.grid(True)
        plt.axis('square')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.show()

def encontrar_trechos(arr: np.ndarray, limiar: float, max_separacao: int, padding: int) -> list:
    """
    Encontra e retorna trechos de um array onde amostras com valor absoluto 
    maior que o limiar estão separadas por no máximo `max_separacao` amostras
    com valor absoluto menor ou igual ao limiar.

    Args:
        arr: O array de entrada do tipo np.complex64.
        limiar: O valor de limiar para o valor absoluto das amostras.
        max_separacao: O número máximo de amostras "abaixo do limiar" permitidas
                       entre amostras "acima do limiar" em um mesmo trecho.

    Returns:
        Uma lista de arrays numpy, onde cada array é um trecho que 
        satisfaz as condições.
    """
    # Passo 1: Encontrar os índices de todas as amostras cujo valor absoluto excede o limiar.
    # np.where retorna uma tupla, então pegamos o primeiro elemento ([0]).
    indices_acima_limiar = np.where(np.abs(arr) > limiar)[0]

    # Se nenhum elemento estiver acima do limiar, retorna uma lista vazia.
    if len(indices_acima_limiar) == 0:
        return []

    # Passo 2: Calcular a diferença entre os índices consecutivos.
    # Isso nos dá o número de amostras entre cada par de amostras "acima do limiar".
    diferencas = np.diff(indices_acima_limiar)

    # Passo 3: Identificar onde os "gaps" (espaços) são maiores que o permitido.
    # Uma diferença maior que 'max_separacao + 1' indica uma quebra de segmento.
    # O '+1' é porque, por exemplo, se os índices são 5 e 8, a diferença é 3, 
    # mas há apenas 2 amostras entre eles (índices 6 e 7).
    # O resultado de np.where são os pontos *onde* ocorrem as quebras.
    pontos_de_quebra = np.where(diferencas > max_separacao + 1)[0]

    # Passo 4: Dividir o array de índices nos pontos de quebra para formar os grupos.
    # np.split divide o array 'indices_acima_limiar' nos locais indicados.
    # O '+1' é necessário porque `pontos_de_quebra` se refere ao índice do array `diferencas`,
    # e queremos dividir *após* esse ponto.
    grupos_de_indices = np.split(indices_acima_limiar, pontos_de_quebra + 1)

    # Passo 5: Usar o primeiro e o último índice de cada grupo para extrair os trechos do array original.
    trechos = [arr[max(0, grupo[0]-padding):min(len(arr)-1, grupo[-1]+padding) + 1] for grupo in grupos_de_indices if grupo.size > 0]

    return trechos

if __name__ == '__main__':
    main()
