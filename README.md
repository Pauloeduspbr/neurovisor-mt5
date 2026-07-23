# NeuroVisor

Porte em C++ (DLL) e MQL5 de um indicador de Machine Learning para MetaTrader 5 — uma rede neural feedforward simples que gera previsão de preço e permite visualizar a curva de perda (MSE) durante o treino.

![NeuroVisor](screenshot.png)

## O que é

Este repositório é um porte técnico do indicador Pine "Machine Learning using Neural Networks | Educational", publicado na TradingView por Alien_Algorithms, reimplementado do zero em C++ e MQL5 (nenhuma linha do script original foi reaproveitada — apenas a lógica).

O motor implementa uma rede neural feedforward de 1 camada oculta com 2 neurônios:

- **Forward pass**: multiplicação de matrizes (pesos `w1` entrada→oculta, `w2` oculta→saída) com ativação sigmoide em cada camada.
- **Treino**: a cada barra confirmada, a rede executa `epochs` iterações de backpropagation sobre o preço normalizado/padronizado, ajustando `w1`/`w2` por gradiente descendente com taxa de aprendizado (`Learning Rate`) configurável. Duas variantes de backprop estão disponíveis — simples e verbosa —, replicando as duas funções equivalentes do script original.
- **Inicialização**: pesos aleatorizados a cada treino, com semente para reprodutibilidade dentro do próprio motor. Ressalva declarada: o gerador `math.random(seed)` do Pine não é uma API pública documentada; o porte usa um LCG (linear congruential generator) equivalente em comportamento — convergência e formato da curva de perda batem com o original —, mas os valores absolutos dos pesos iniciais não são reproduzíveis bit a bit contra o script Pine. É uma limitação de plataforma, não do porte.
- **Saída**: a previsão de preço da rede é plotada sobre o preço real; opcionalmente, a curva de MSE ao longo das épocas de treino é plotada em painel separado.

A DLL é stateless: os pesos (`w1`/`w2`) cruzam a fronteira C++↔MQL5 por parâmetro (`double*`, entrada e saída) e vivem em buffer do lado MQL5, por instância de gráfico — o que evita que duas instâncias do indicador (dois gráficos ou símbolos) compartilhem o mesmo estado de treino.

Uso pretendido: estudo da mecânica de uma rede neural (forward pass, backpropagation, gradiente, curva de perda) aplicada a dados de preço — não é um gerador de sinal de entrada/saída pronto para uso.

## Instalação — versão pré-compilada

1. Copie `neural_net.dll` para a pasta `MQL5/Libraries` do seu terminal MetaTrader 5.
2. Copie `TV_04_NeuralNet.ex5` para a pasta `MQL5/Indicators`.
3. Reinicie o MetaTrader 5 (ou, no Navigator, clique com o botão direito em "Indicadores" e atualize a lista).
4. Arraste o indicador `TV_04_NeuralNet` para o gráfico desejado.

## Build a partir do código-fonte

1. Compile o C++ (`neural_net`) com g++/MinGW-w64, usando o `build.sh` incluso em `src/cpp/`. Isso gera `neural_net.dll` (x64).
2. Abra `src/mql5/TV_04_NeuralNet.mq5` no MetaEditor (integrado ao MetaTrader 5).
3. Compile com F7 para gerar o `.ex5`.
4. Copie os artefatos gerados (`neural_net.dll` e `TV_04_NeuralNet.ex5`) para as pastas do terminal conforme os passos de instalação acima.

## Licença

Este repositório é licenciado sob CC-BY-NC-SA-4.0; a lógica original em Pine Script é de autoria de Alien_Algorithms (TradingView), aqui reimplementada do zero em C++ e MQL5.

## Aviso

Uso educacional e de análise técnica. Não constitui recomendação de investimento.
