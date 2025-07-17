"""Implementação do código convolucional binário IEEE 802.11 taxa 1/2.

Referências: IEEE 802.11a, Seção 17.3.5.5; Schwarzinger, Seções 5.6.2,
5.6.3 e 5.6.4; gr-ieee802-11, `lib/viterbi_decoder/base.h`,
`lib/viterbi_decoder/viterbi_decoder_generic.cc` e
`lib/viterbi_decoder/viterbi_decoder_x86.cc`.

Convenção do registrador de deslocamento: segue o codificador de apoio
`BinaryConvolutionalEncoder.py`, do livro Digital Signal Processing in Modern
Communication Systems, de Schwarzinger, arquivado em
https://web.archive.org/web/20240614143326/http://signal-processing.net/Python/BinaryConvolutionalEncoder.py:
o bit de entrada mais recente ocupa o MSB do registrador de comprimento de
restrição 7 antes da aplicação dos geradores octais 133 e 171.
"""

import numpy as np
from numba import njit

CONSTRAINT_LENGTH = 7
GENERATORS = (0o133, 0o171)
NUM_STATES = 1 << (CONSTRAINT_LENGTH - 1)
_NEG_INF = -1.0e300


def _parity(value):
    return value.bit_count() & 1


def _make_transition_tables():
    """Pré-calcula a treliça do código convolucional.

    `next_state[state, input_bit]` guarda o destino do ramo. `expected` guarda
    os dois bits codificados esperados já no mapeamento bipolar usado pelo
    Viterbi soft-decision. Pré-calcular isso evita refazer paridade dentro do
    laço crítico compilado pelo Numba.
    """
    next_state = np.zeros((NUM_STATES, 2), dtype=np.int64)
    expected = np.zeros((NUM_STATES, 2, 2), dtype=np.float64)

    for state in range(NUM_STATES):
        for input_bit in (0, 1):
            register = (input_bit << (CONSTRAINT_LENGTH - 1)) | state
            next_state[state, input_bit] = (
                (input_bit << (CONSTRAINT_LENGTH - 2)) | (state >> 1)
            ) & (NUM_STATES - 1)
            expected[state, input_bit, 0] = 2 * _parity(register & GENERATORS[0]) - 1
            expected[state, input_bit, 1] = 2 * _parity(register & GENERATORS[1]) - 1

    return next_state, expected


NEXT_STATE_TABLE, EXPECTED_TABLE = _make_transition_tables()


def encode_bits(bits):
    """Codifica uma sequência de bits com o código IEEE 802.11 K=7, taxa 1/2."""
    state = 0
    output = []

    for bit in bits:
        input_bit = int(bit) & 1
        register = (input_bit << (CONSTRAINT_LENGTH - 1)) | state
        output.append(_parity(register & GENERATORS[0]))
        output.append(_parity(register & GENERATORS[1]))
        state = ((input_bit << (CONSTRAINT_LENGTH - 2)) | (state >> 1)) & (NUM_STATES - 1)

    return output


@njit(cache=True)
def _decode_soft_numba(soft, next_state_table, expected_table):
    """Núcleo Viterbi soft-decision compilado com Numba.

    Args:
        soft: Vetor unidimensional com dois soft bits por bit de informação.
        next_state_table: Tabela de transição da treliça.
        expected_table: Bits codificados esperados em formato bipolar.

    Returns:
        Bits de informação estimados como `uint8`, um bit por passo da treliça.

    A métrica de ramo é a correlação entre os soft bits recebidos e os bits
    codificados esperados no mapeamento bipolar 0 -> -1 e 1 -> +1. Por isso,
    a melhor métrica acumulada é a maior. O traceback começa no estado final
    mais provável; os bits de cauda do IEEE 802.11 são verificados depois pelo
    receptor.
    """
    num_steps = soft.shape[0] // 2
    metrics = np.empty(NUM_STATES, dtype=np.float64)
    next_metrics = np.empty(NUM_STATES, dtype=np.float64)
    survivors = np.zeros((num_steps, NUM_STATES), dtype=np.uint8)

    # Invariante inicial da treliça: o codificador IEEE 802.11 parte do estado
    # zero. Os demais estados começam inviáveis e só passam a competir quando
    # algum caminho válido chega até eles.
    for state in range(NUM_STATES):
        metrics[state] = _NEG_INF
    metrics[0] = 0.0

    for step in range(num_steps):
        # Cada passo de treliça consome os dois bits codificados produzidos por
        # um único bit de entrada do codificador taxa 1/2.
        r0 = soft[2 * step]
        r1 = soft[2 * step + 1]

        # `next_metrics` representa exclusivamente o instante atual. Reiniciar
        # todos os estados evita carregar sobreviventes de passos anteriores.
        for state in range(NUM_STATES):
            next_metrics[state] = _NEG_INF

        for state in range(NUM_STATES):
            path_metric = metrics[state]
            if path_metric <= _NEG_INF / 2:
                continue

            for input_bit in range(2):
                destination = next_state_table[state, input_bit]

                # Métrica soft-decision: correlação dos dois soft bits recebidos
                # com a saída esperada do ramo. A magnitude do soft bit pesa a
                # confiança; quantizar antes desta soma transformaria o algoritmo
                # em hard-decision.
                # TAREFA DO ALUNO: substitua o valor abaixo pela métrica de ramo.
                branch_metric = 0.0
                candidate_metric = path_metric + branch_metric

                # Operação ACS (add-compare-select): se dois caminhos chegam ao
                # mesmo destino, apenas o de maior métrica sobrevive.
                # TAREFA DO ALUNO: compare `candidate_metric` com a métrica atual
                # do destino, atualize `next_metrics` e grave o sobrevivente.

        # Avança a treliça: as métricas recém-calculadas tornam-se as métricas
        # anteriores do próximo passo.
        for state in range(NUM_STATES):
            metrics[state] = next_metrics[state]

    # Esta implementação escolhe o estado final mais provável. Em seguida,
    # `decode_data_symbols` verifica os bits de cauda para diagnosticar se o
    # caminho terminado é compatível com o zeramento exigido pelo IEEE 802.11.
    best_state = 0
    # TAREFA DO ALUNO: escolha o estado final com a maior métrica acumulada.

    decoded = np.zeros(num_steps, dtype=np.uint8)

    # No traceback, o bit que entrou no codificador é o MSB do estado de destino,
    # porque a convenção deste código desloca o registrador para a direita.
    for step in range(num_steps - 1, -1, -1):
        # TAREFA DO ALUNO: antes de avançar para o estado anterior, grave em
        # `decoded[step]` o MSB do estado de destino atual.
        best_state = survivors[step, best_state]

    return decoded


def decode_soft(soft_bits):
    """Decodifica soft bits com Viterbi usando métrica de ramo por correlação.

    `soft_bits` deve conter dois valores por bit de entrada. Valores positivos
    favorecem bit codificado 1, valores negativos favorecem bit codificado 0,
    e a magnitude representa a confiança. A treliça de Viterbi segue
    Schwarzinger, Seção 5.6.4; em cada ramo, esta implementação maximiza a
    métrica soft-decision bipolar equivalente `sum(r_i * expected_bipolar_i)`.
    A função é mantida pequena para que o trabalho algorítmico fique concentrado
    em `_decode_soft_numba`, que também será o ponto preenchido no molde.
    """
    soft = np.asarray(soft_bits, dtype=np.float64).ravel()
    if soft.size % 2 != 0:
        raise ValueError("soft_bits length must be even for the rate-1/2 code")

    return _decode_soft_numba(soft, NEXT_STATE_TABLE, EXPECTED_TABLE)
