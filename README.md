# Implementação de um Transceptor IEEE 802.11a/g em Python

Nesta prática, vamos implementar um receptor completo para a camada física (PHY) do padrão de rede sem fio IEEE 802.11a/g. A lógica será capaz de processar um sinal I/Q complexo, sincronizar com os pacotes recebidos, estimar e corrigir distorções do canal, e decodificar os símbolos OFDM para extrair os bits de dados originais.

O projeto é baseado nos conceitos e algoritmos descritos no livro-texto e na norma IEEE 802.11a, e visa fornecer uma compreensão prática e aprofundada dos blocos de processamento de sinais em um modem OFDM moderno. Procuramos implementar a maior parte do código como uma tradução direta para Python do código MATLAB que acompanha o livro, facilitando a consulta ao livro e a comparação entre implementações.

## Referências

*   **Livro-texto:** SCHWARZINGER, Andreas. Digital Signal Processing in Modern Communication Systems. 2. ed. 2022. 637 p. ISBN 9780988873506.
*   **Norma IEEE 802.11a:** [https://gtas.unican.es/files/docencia/CD/practicas/ieee_802.11a.pdf](https://gtas.unican.es/files/docencia/CD/practicas/ieee_802.11a.pdf)
*   **Implementação de Referência (GNU Radio):** [https://github.com/bastibl/gr-ieee802-11](https://github.com/bastibl/gr-ieee802-11)

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

As funções não precisam ser implementadas em nenhuma ordem particular, mas a ordem abaixo acompanha a lógica de funcionamento do modem:

1.  **`rx.packet_detector`**: Implementa a detecção de pacotes usando a autocorrelação da sequência de treinamento curta.
    *   Teste com: `poetry run python -m unittest tests.test_packet_detector`

2.  **`rx.detect_frequency_offsets`**: Estima os deslocamentos de frequência grosseiro e fino usando as sequências de treinamento curta e longa.
    *   Teste com: `poetry run python -m unittest tests.test_freq_offset`

3.  **`rx.long_symbol_correlator`**: Encontra a posição exata do símbolo de treinamento longo para sincronização fina de tempo.
    *   Teste com: `poetry run python -m unittest tests.test_long_symbol_corr`

4.  **`rx.ofdm_receiver`**: Implementa o laço principal do receptor OFDM, que orquestra as etapas de sincronização e processa cada símbolo. A maior parte dessa função já está implementada, mas faltam dois pontos cuja compreensão é bastante importante:
    *   **Correção de Deriva de Fase:** Utilizando os quatro tons piloto de cada símbolo, calcule o erro de fase médio (`theta`) e aplique a correção para desfazer a rotação da constelação.
    *   **Correção de Deriva de Temporização:** Calcule a inclinação (`slope`) da fase através dos tons piloto e use-a para corrigir o erro de fase linear causado pela deriva de temporização.
    *   Teste com: `poetry run python -m unittest tests.test_ofdm_receiver`

5.  **`rx.demapper_ofdm`**: Converte os símbolos complexos recebidos em "soft bits" (LLRs).
    *   Teste com: `poetry run python -m unittest tests.test_demapper`

6.  **`rx.create_deinterleaving_pattern`**: Gera o padrão para reverter o processo de entrelaçamento de bits.
    *   Teste com: `poetry run python -m unittest tests.test_deinterleaver`

7.  **`rx.convolutional_decoder`**: Implementa o decodificador Viterbi para correção de erros.
    *   Teste com: `poetry run python -m unittest tests.test_viterbi`

8.  **`rx.descramble`**: Reverte o processo de embaralhamento para recuperar os dados originais.
    *   Teste com: `poetry run python -m unittest tests.test_descramble`

Após todas as funções acima estarem implementadas e seus respectivos testes passarem, o teste de integração final, que executa o receptor completo, deve passar:

```bash
poetry run python -m unittest tests.test_integration
```

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

1.  **Gere Sinais Wi-Fi:** Em um computador Linux com uma placa de rede sem fio compatível, você pode criar um ponto de acesso (Access Point) usando `hostapd`. Um arquivo de configuração de exemplo, `workbench/hostapd.conf`, é fornecido. Ele configura um AP para transmitir beacons 802.11g (que usam a mesma camada física do 802.11a) a uma taxa de 12 Mbps, fornecendo um sinal simples e conhecido para teste.

2.  **Capture o Sinal I/Q:** Use o GNU Radio Companion com o arquivo de fluxo fornecido, `workbench/wifi_rx.grc`. Este fluxo permite que você:
    *   Configure seu SDR para receber no canal Wi-Fi correto.
    *   Visualize o espectro e a constelação em tempo real usando a implementação de referência `gr-ieee802-11` para verificar se a captura está funcionando.
    *   Salve as amostras I/Q brutas em um arquivo (por exemplo, `/tmp/wifi_capture.iq`).

3.  **Processe com seu Receptor:** Execute seu script no modo `--iq`, passando o arquivo que você acabou de gravar:
    ```bash
    poetry run python -m ieee80211ag --iq /tmp/wifi_capture.iq
    ```

Se sua implementação estiver correta, você deverá ver os quadros de beacon decodificados impressos no console.
