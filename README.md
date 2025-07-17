# Implementação de um Transceptor IEEE 802.11a/g em Python

Nesta prática, vamos implementar um receptor completo para a camada física (PHY) do padrão de rede sem fio IEEE 802.11a/g. A lógica será capaz de processar um sinal I/Q complexo, sincronizar com os pacotes recebidos, estimar e corrigir distorções do canal, e decodificar os símbolos OFDM para extrair os bits de dados originais.

O projeto é baseado nos conceitos e algoritmos descritos no livro-texto e na norma IEEE 802.11a, e visa fornecer uma compreensão prática e aprofundada dos blocos de processamento de sinais em um modem OFDM moderno. Procuramos implementar a maior parte do código como uma tradução direta para Python do código MATLAB que acompanha o livro, facilitando a consulta ao livro e a comparação entre implementações.

## Referências

*   **Livro-texto:** SCHWARZINGER, Andreas O. *Digital Signal Processing in Modern Communication Systems*. 2. ed. Orlando, FL: Andreas O. Schwarzinger, 2022. 637 p. ISBN 978-0-9888735-0-6. [Amazon](https://www.amazon.com/s?k=9780988873506).
*   **Norma IEEE 802.11a:** IEEE Computer Society. *IEEE Std 802.11a-1999: Supplement to IEEE Standard for Information Technology--Telecommunications and Information Exchange Between Systems--Local and Metropolitan Area Networks--Specific Requirements--Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications--High-speed Physical Layer in the 5 GHz Band*. New York: IEEE, 1999. [PDF](https://gtas.unican.es/files/docencia/CD/practicas/ieee_802.11a.pdf).
*   **Implementação de referência em GNU Radio:** BLOESSL, Bastian et al. *gr-ieee802-11: IEEE 802.11 a/g/p Transceiver*. [GitHub](https://github.com/bastibl/gr-ieee802-11).

## Dependências e Configuração

Este projeto utiliza [Python Poetry](https://python-poetry.org/) para gerenciar suas dependências e ambiente virtual.

### Instalação do Poetry

*   **Ubuntu:**
    ```bash
    sudo apt update && sudo apt install python3-poetry
    ```

*   **Arch Linux:**
    ```bash
    sudo pacman -S python-poetry
    ```

*   **Windows (via PowerShell):**
    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

### Instalação das Dependências do Projeto

Após instalar o Poetry, navegue até o diretório raiz deste projeto e execute o seguinte comando para instalar todas as bibliotecas necessárias em um ambiente virtual dedicado:

```bash
poetry install
```

## Implementação

As partes do código que estão faltando estão localizadas no diretório `ieee80211ag/` e marcadas com o comentário `# TAREFA DO ALUNO`. Cada função concluída permite que um novo teste unitário passe, validando seu trabalho passo a passo.

Para executar os testes unitários, utilize o comando:

```bash
poetry run python -m unittest discover tests
```

Para executar a mesma correção automática usada no GitHub Classroom, use:

```bash
./run-grader
```

As funções não precisam ser implementadas em nenhuma ordem particular, mas a ordem abaixo acompanha a lógica de funcionamento do modem e tende a facilitar a depuração.

### Roteiro de Implementação

1.  **`rx.packet_detector`**: detecta a STS pela autocorrelação com atraso de 16 amostras.

    O código deve manter três buffers deslizantes: atraso de 16 amostras, média móvel da autocorrelação e média móvel da potência. Para cada amostra, calcule `rx_input[i] * conj(rx_input_16)`, suavize esse valor, normalize pela potência média local e aplique a histerese 0.85/0.65. A saída mais importante para as próximas etapas é `falling_edge_position`, a borda de descida do indicador de pacote.

    O teste `tests.test_packet_detector` verifica se a métrica, a flag de detecção e a borda encontrada são coerentes com um pacote sintético.

2.  **`rx.detect_frequency_offsets`**: estima o deslocamento de frequência grosseiro e fino.

    A estimativa grosseira usa a fase da autocorrelação da STS com atraso 16; a estimativa fina usa a fase da autocorrelação da LTS com atraso 64. Em ambos os casos, use `Delta_f = theta * Fs / (2*pi*D)`, com `Fs = 20e6` e `D` igual ao atraso em amostras. Meça a fase em pontos estáveis das sequências de treinamento, não exatamente na borda de transição.

    A referência `falling_edge_position` é a borda de descida do detector baseado na STS, portanto ela fica próxima da transição entre a sequência curta e a região da LTS, mas não é uma fronteira ideal do preâmbulo. A STS tem 10 repetições de 16 amostras, e a LTS tem `GI2 + 2*64` amostras. Na prática, use uma amostra no platô da STS, cerca de 50 amostras antes da borda, para a estimativa grosseira; para a estimativa fina, use uma amostra já dentro da segunda repetição longa, cerca de 125 amostras depois da borda. Sempre verifique se o índice escolhido está dentro do vetor antes de acessar a autocorrelação.

    O teste `tests.test_freq_offset` verifica as duas estimativas em sinais com erro de frequência conhecido.

3.  **`rx.long_symbol_correlator`**: faz a sincronização fina de tempo pela LTS.

    Construa um correlacionador FIR com a LTS invertida e conjugada. Nesta prática, a LTS local é quantizada pelo sinal das componentes I/Q, como no código MATLAB do livro. Procure o maior pico apenas na janela plausível depois da borda grosseira para evitar falsos picos fora da região do preâmbulo. Uma janela de uma LTS começando cerca de 54 amostras depois de `falling_edge_position` é suficiente para os sinais da prática.

    O teste `tests.test_long_symbol_corr` verifica se o pico da correlação aparece na posição esperada.

4.  **`rx.ofdm_receiver`**: processa os símbolos OFDM após sincronização e equalização.

    A maior parte da função organiza o receptor completo. Os pontos principais a preencher no molde são a correção de fase comum e a correção de inclinação de fase por símbolo:

    ```python
    pilots = equalized_symbol[PILOT_CARRIERS_IDX] * polaridade_esperada
    averaged_pilot = soma_ponderada_dos_pilotos
    theta = angle(averaged_pilot)
    corrected_symbol1 = equalized_symbol * exp(-1j * theta)
    slope = soma_ponderada(angle(pilots) / PILOT_CARRIERS)
    corrected_symbol2 = corrected_symbol1 * exp(-1j * k * average_slope)
    ```

    O vetor `pilots` deve estar na ordem das subportadoras `[-21, -7, 7, 21]`. A fase comum corrige uma rotação igual para todas as subportadoras; a inclinação corrige o erro residual de temporização, que aparece como uma rampa de fase em função da frequência.

    O teste `tests.test_ofdm_receiver` verifica se os símbolos corrigidos ficam próximos dos símbolos transmitidos em uma simulação controlada.

5.  **`rx.demapper_ofdm`**: converte símbolos complexos em soft bits.

    Para BPSK, retorne a parte real de cada símbolo. Para QPSK, intercale parte real e imaginária na ordem `[I0, Q0, I1, Q1, ...]`; uma forma direta é empilhar os vetores I/Q por coluna e depois achatar o resultado. Não quantize esses valores para 0 ou 1: o sinal do soft bit apenas indica qual bit codificado é favorecido, e a magnitude indica a confiança usada pelo Viterbi.

    O teste `tests.test_demapper` verifica BPSK, QPSK e o erro esperado para modulações não implementadas.

6.  **`rx.create_deinterleaving_pattern`**: gera a permutação inversa do interleaver.

    A norma IEEE 802.11a define duas permutações para espalhar bits codificados entre subportadoras. O receptor precisa desfazer essas permutações por símbolo OFDM. Calcule a primeira permutação em função do índice original `k`, construa o mapa da segunda permutação e combine os dois mapas para obter o vetor que reordena os soft bits recebidos para a ordem anterior ao interleaving.

    As equações usadas pelo transmissor são `i = (N_CBPS // 16) * (k % 16) + (k // 16)`, `s = max(N_BPSC // 2, 1)` e `j = s * (i // s) + (i + N_CBPS - (16*i // N_CBPS)) % s`. A função `create_interleaving_pattern()` em `ieee80211ag/tx.py` implementa o caminho direto. Compare com ela: o `np.argsort` usado no transmissor inverte o mapa direto para permitir `bits[pattern]`; no receptor, o padrão deve desfazer essa operação quando aplicado ao vetor recebido.

    O teste `tests.test_deinterleaver` verifica o padrão para os valores de `N_CBPS` e `N_BPSC` usados na prática.

7.  **`bcc._decode_soft_numba`**: implementa o núcleo do decodificador Viterbi com soft-decision.

    A função `rx.convolutional_decoder` apenas chama esse núcleo. A estrutura ao redor já fornece os parâmetros do código convolucional IEEE 802.11 de taxa 1/2 e comprimento de restrição 7:

    *   `NEXT_STATE_TABLE[state, input_bit]`: próximo estado da treliça.
    *   `EXPECTED_TABLE[state, input_bit, :]`: dois bits codificados esperados para aquele ramo, no mapeamento bipolar `0 -> -1` e `1 -> +1`.
    *   `metrics`: métrica acumulada dos caminhos sobreviventes no instante anterior.
    *   `next_metrics`: métrica acumulada dos caminhos candidatos no instante atual.
    *   `survivors[step, state]`: estado anterior escolhido como sobrevivente para chegar a `state` naquele passo.

    A cada passo da treliça, leia o par de soft bits recebido, teste os dois ramos que saem de cada estado alcançável e calcule a métrica de ramo por correlação:

    ```python
    branch_metric = r0 * expected0 + r1 * expected1
    ```

    Como este é um decodificador soft-decision, a melhor métrica é a maior, não a menor. Ao encontrar uma métrica melhor para o estado de destino, atualize a métrica acumulada e grave o estado anterior em `survivors`. No fim, escolha o estado final com a maior métrica acumulada e faça o traceback a partir dele. Os quadros 802.11 usados nesta prática incluem seis bits de cauda zerados; nesta implementação, esses bits são verificados depois como diagnóstico (`tail_ok`), em vez de serem usados para forçar o estado inicial do traceback.

    Não implemente uma versão alternativa em Python puro para contornar o Numba. O objetivo é que o laço principal seja compatível com compilação `nopython`: use arrays NumPy já alocados, laços explícitos e escalares simples. Evite listas Python, dicionários, objetos dinâmicos e chamadas que o Numba não consiga compilar. Se a solução usar recursos incompatíveis com esse modelo de implementação eficiente, a própria compilação deve apontar o problema.

    O teste `tests.test_viterbi` verifica o codificador, a decodificação sob ruído e casos em que a informação de confiabilidade dos soft bits é necessária.

8.  **`rx.descramble`**: reverte o scrambler do campo DATA.

    Os sete primeiros bits do SERVICE são zero antes do scrambling. Depois do Viterbi, esses sete bits recebidos revelam o estado inicial do LFSR. Reconstrua esse estado, aplique a mesma função `scramble` sobre os bits restantes e substitua os sete primeiros bits por zeros. Isso funciona porque o scrambler é uma sequência pseudoaleatória combinada por XOR, e o XOR é sua própria operação inversa.

    Atenção à ordem dos bits: `scramble(..., initial_state)` interpreta `initial_state` com `format(initial_state, '07b')`, isto é, MSB primeiro. Os sete bits observados no início do SERVICE chegam na ordem de transmissão; por isso, antes de convertê-los para inteiro, alinhe-os com a ordem esperada pelo estado interno do LFSR.

    O teste `tests.test_descramble` verifica se um fluxo embaralhado volta aos bits originais.

Após todas as funções acima estarem implementadas e seus respectivos testes passarem, o teste de integração final, que executa o receptor completo, deve passar:

```bash
poetry run python -m unittest tests.test_integration
```

## Avaliação

A nota total da prática é 10 pontos. A correção automática vale 8 pontos e é
executada pelo GitHub Classroom com os seguintes pesos:

| Item | Pontos |
| --- | ---: |
| Detecção de pacote | 0,75 |
| Estimativa de deslocamento de frequência | 0,75 |
| Correlação do símbolo longo | 0,75 |
| Receptor OFDM | 1,25 |
| Demapper OFDM | 0,75 |
| Deinterleaver | 0,50 |
| Viterbi | 1,25 |
| Descrambler | 0,50 |
| Integração | 1,50 |

Os 2 pontos restantes são atribuídos no teste de bancada com SDR.

## Uso e Depuração

O script principal `ieee80211ag/__main__.py` serve como uma ferramenta poderosa para executar o transceptor completo e depurar sua implementação. Ele suporta três modos de operação:

### Modo Testbench (`--testbench`)

Este modo executa uma simulação completa:
1.  Um transmissor gera um pacote 802.11a sintético.
2.  O pacote passa por um modelo de canal que simula defeitos do mundo real (ruído, múltiplos percursos, etc.).
3.  O receptor processa o sinal distorcido.
4.  O desempenho é avaliado e vários gráficos são gerados, incluindo:
    *   **EVM vs. Frequência:** Mostra como a qualidade do sinal varia entre as diferentes subportadoras. É muito útil para visualizar os efeitos do fading seletivo de múltiplos percursos.
    *   **EVM vs. Tempo:** Mostra a qualidade do sinal símbolo a símbolo. É ideal para identificar problemas com ruído de fase ou rastreamento de tempo.
    *   **Constelação:** Visualiza os símbolos recebidos após a equalização.

Exemplo de uso:
```bash
poetry run python -m ieee80211ag --testbench
```

### Modo de Arquivo NPZ (`--npz`)

Processa um arquivo `.npz` contendo trechos de sinais I/Q pré-gravados. Um arquivo de exemplo, `tests/data/signal.npz`, é fornecido e pode ser usado para depurar o receptor com dados consistentes.

Exemplo de uso:
```bash
poetry run python -m ieee80211ag --npz tests/data/signal.npz
```

### Modo de Arquivo I/Q (`--iq`)

Processa um arquivo I/Q bruto (formato `complex64`) capturado com um Software Defined Radio (SDR). Este modo é ideal para testar seu receptor com sinais Wi-Fi reais.

Exemplo de uso:
```bash
poetry run python -m ieee80211ag --iq /tmp/wifi_capture.iq
```

## Teste de Bancada com SDR

Para um teste completo com hardware real, você pode usar um SDR (como um LimeSDR) para capturar sinais Wi-Fi e processá-los com seu código.

1.  **Gere Sinais Wi-Fi:** Em um computador Linux com uma placa de rede sem fio compatível, você pode criar um ponto de acesso (Access Point) usando `hostapd`. Um arquivo de configuração de exemplo, `workbench/hostapd.conf`, é fornecido. Ele configura um AP para transmitir beacons 802.11g em modo OFDM a 12 Mbps; esse modo usa os mesmos blocos de modulação OFDM estudados nesta prática, embora opere na faixa de 2,4 GHz.

2.  **Capture o Sinal I/Q:** Use o GNU Radio Companion com o arquivo de fluxo fornecido, `workbench/wifi_rx.grc`. Antes de abrir esse fluxo, instale o `gr-ieee802-11` no mesmo ambiente do GNU Radio; caso contrário, o GRC não encontrará os blocos usados para visualizar e conferir a captura. Arch Linux: pacote `gr-ieee802-11-git` no [Chaotic-AUR](https://aur.chaotic.cx/).

    Este fluxo permite que você:
    *   Configure seu SDR para receber no canal Wi-Fi correto.
    *   Visualize o espectro e a constelação em tempo real usando a implementação de referência `gr-ieee802-11` para verificar se a captura está funcionando.
    *   Salve as amostras I/Q brutas em um arquivo (por exemplo, `/tmp/wifi_capture.iq`).

3.  **Processe com seu Receptor:** Execute seu script no modo `--iq`, passando o arquivo que você acabou de gravar:
    ```bash
    poetry run python -m ieee80211ag --iq /tmp/wifi_capture.iq
    ```

Se sua implementação estiver correta, você deverá ver os quadros de beacon decodificados impressos no console.
