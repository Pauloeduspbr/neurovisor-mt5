//+------------------------------------------------------------------+
//| neural_net.h                                                     |
//| Engine C++ do "Neural Network Showcase"                          |
//| (c) Alien_Algorithms — CC BY-NC-SA 4.0 (NAO comercial), Pine v5   |
//|                                                                  |
//| Porte de:                                                        |
//|   indicadores_tradingview/04_Machine_Learning_using_Neural_...    |
//|   source.pine                                                    |
//|                                                                  |
//| LICENCA: Attribution-NonCommercial-ShareAlike 4.0. Uso pessoal e  |
//| estudo livres; uso COMERCIAL nao permitido pelo autor.            |
//|                                                                  |
//| ARQUITETURA (fiel ao original)                                    |
//|   rede 2-2-1, ativacao sigmoid nas duas camadas, perda MSE        |
//|   entrada : [ close[1]/maxScale , close[2]/maxScale ]             |
//|   alvo    : close/maxScale                                        |
//|   maxScale: maior high visto ate a barra (normalizacao do autor)  |
//|   treino  : `epochs` iteracoes, SO na ultima barra confirmada     |
//|                                                                  |
//| ABI 2 — DLL SEM ESTADO (mudanca de contrato, ver nota abaixo)     |
//| Os pesos entram e saem por parametro; a DLL nao guarda nada entre |
//| chamadas. Isso segue o padrao dos demais indicadores do lote.     |
//| A versao ABI 1 guardava os pesos em variaveis globais e tinha     |
//| DOIS defeitos medidos:                                            |
//|   (a) ACUMULACAO — o wrapper chama NnTrain a cada barra nova e o  |
//|       treino continuava de onde parou. Medido em EURUSD H1: apos  |
//|       20 barras, loss=1.8e-11 e Predicted COLADO no Actual (0.0   |
//|       pips), com a curva de perda achatada. O Pine treina UMA vez |
//|       (`barstate.islastconfirmedhistory`), entao o porte divergia.|
//|   (b) COMPARTILHAMENTO — dois graficos com o indicador treinavam  |
//|       a MESMA rede com dados diferentes.                          |
//|                                                                  |
//| TRES PARTICULARIDADES DO ORIGINAL, PRESERVADAS                    |
//| 1) backpropagation_simple (linhas 77-86) NAO aplica a derivada da |
//|    sigmoid ao atualizar w1 — esta matematicamente errado, e e     |
//|    exatamente o contraste didatico que o indicador quer mostrar.  |
//|    Mantido como modo alternativo.                                 |
//| 2) Nos DOIS modos, w1 e atualizado lendo o w2 JA ATUALIZADO na    |
//|    mesma iteracao (linha 96 escreve, linha 102 le). O correto     |
//|    seria usar o w2 anterior. Replicado como esta.                 |
//| 3) `finalLoss` e a perda da ULTIMA EPOCA, medida ANTES do backprop|
//|    final, enquanto `predicted` vem do feedforward DEPOIS dele.    |
//|    Logo o rotulo "MSE Loss" descreve um estado ligeiramente       |
//|    anterior ao desenhado (medido: 8.8 pips de 463). NAO e bug do  |
//|    porte: o Pine exibe array.get(loss_curve, epochs-1), linha 160.|
//|                                                                   |
//| COMO LER O MSE (fonte recorrente de confusao)                      |
//| A perda e calculada no espaco NORMALIZADO e e o erro AO QUADRADO. |
//| Para converter em preco:  erro = sqrt(loss) * maxScale.           |
//| Medido em EURUSD H1 (maxScale=1.12756): MSE 1.59e-05 = 45 pips.   |
//| Um MSE "de aparencia minuscula" NAO significa previsao precisa.   |
//|                                                                  |
//| LIMITE DECLARADO: math.random(0,1,seed) do Pine                   |
//| O gerador pseudoaleatorio do TradingView nao e publico, entao os  |
//| PESOS INICIAIS nao podem ser reproduzidos bit a bit. Usamos um    |
//| LCG deterministico com a mesma semente (1337). Consequencia: a    |
//| curva de perda e o valor previsto diferem em magnitude dos do     |
//| TradingView, embora o comportamento (convergencia, formato da     |
//| curva) seja o mesmo. Isso e limitacao da plataforma, nao do porte.|
//+------------------------------------------------------------------+
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define NN_ABI_VERSION 2
#define NN_WEIGHTS 6      /* w1[0][0] w1[0][1] w1[1][0] w1[1][1] w2[0] w2[1] */

__declspec(dllexport) int __stdcall NnVersion(void);

//+------------------------------------------------------------------+
//| NnInitWeights — sorteia os NN_WEIGHTS pesos iniciais              |
//| Equivale ao fillMatrixRandomly do Pine com math.random(0,1,seed). |
//| Retorna quantos escreveu, ou -1.                                  |
//+------------------------------------------------------------------+
__declspec(dllexport) int __stdcall NnInitWeights(double *weights, int count, int seed);

//+------------------------------------------------------------------+
//| NnTrain — treina a rede na barra `atBar` e devolve o resultado    |
//|                                                                   |
//| ENTRADA                                                           |
//|   close[]   serie de fechamento (indice 0 = barra mais antiga)    |
//|   high[]    serie de maxima, para o maxScale do autor             |
//|   size      tamanho das series                                    |
//|   atBar     barra onde treinar (o Pine usa a ultima confirmada)   |
//|                                                                   |
//| PESOS (in/out) — a DLL nao os guarda                              |
//|   weights[NN_WEIGHTS] entram como estado inicial e saem treinados.|
//|   initWeights != 0 -> sorteia antes de treinar (comportamento     |
//|   FIEL: cada barra e uma sessao de treino independente, como o    |
//|   Pine, que dispara uma unica vez). initWeights == 0 -> continua  |
//|   o treino a partir dos pesos recebidos (modo acumulativo, que    |
//|   NAO e o do original — converge para gap zero em ~20 barras).    |
//|                                                                   |
//| PARAMETROS                                                        |
//|   lr, epochs, simpleBackprop (0/1), seed                          |
//|                                                                   |
//| SAIDA                                                             |
//|   lossCurve   perda de cada epoca (ate lossCapacity valores)      |
//|   predicted   preco previsto, ja desnormalizado                   |
//|   actual      preco alvo, ja desnormalizado                       |
//|   finalLoss   perda da ultima epoca (ver particularidade 3)       |
//|   maxScale    o divisor usado; necessario para converter o MSE    |
//|               em preco. Pode ser NULL se nao interessar.          |
//|                                                                   |
//| Retorno: numero de epocas gravadas; -1 em argumento invalido.     |
//+------------------------------------------------------------------+
__declspec(dllexport) int __stdcall NnTrain(
   const double *close, const double *high, int size, int atBar,
   double lr, int epochs, int simpleBackprop, int seed,
   double *weights, int weightsCount, int initWeights,
   double *lossCurve, int lossCapacity,
   double *predicted, double *actual, double *finalLoss, double *maxScale);

#ifdef __cplusplus
}
#endif
