//+------------------------------------------------------------------+
//| neural_net.cpp                                                   |
//| Porte C++ do "Neural Network Showcase" (c) Alien_Algorithms      |
//| CC BY-NC-SA 4.0                                                  |
//|                                                                  |
//| Mapa Pine -> C++:                                                |
//|   linhas 42-55    sigmoid/mse/normalize      -> helpers          |
//|   linhas 59-74    feedforward                -> feedforward()    |
//|   linhas 77-86    backpropagation_simple     -> backprop(simple) |
//|   linhas 88-106   backpropagation_verbose    -> backprop(verbose)|
//|   linhas 110-129  train_nn                   -> NnTrain()        |
//|                                                                  |
//| ABI 2: sem estado global. Os pesos vivem no chamador (buffer do  |
//| MQL5), como nos demais indicadores do lote. Ver nota no .h.      |
//+------------------------------------------------------------------+
#include "neural_net.h"
#include <cmath>

namespace {

//--- os pesos sao uma VIEW sobre o array do chamador, nao copia
struct Net
{
   double *w1[2];      // aponta para weights[0..1] e weights[2..3]
   double *w2;         // aponta para weights[4..5]

   explicit Net(double *w) { w1[0] = w; w1[1] = w + 2; w2 = w + 4; }
};

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double mse(double p, double a) { return (p - a) * (p - a); }

//--- LCG deterministico. Substitui math.random(0,1,seed) do Pine, cujo
//    gerador nao e publico (ver nota no .h).
double lcg01(unsigned int s)
{
   s = 1664525u * s + 1013904223u;
   s = 1664525u * s + 1013904223u;   // dois passos: evita correlacao entre
   return (double)(s >> 8) / 16777216.0;   // sementes proximas (i + j*rows)
}

//--- Pine feedforward(): 2 -> 2 (sigmoid) -> 1 (sigmoid)
void feedforward(const Net &n, const double in[2], double &out, double hidden[2])
{
   for(int i = 0; i < 2; ++i)
   {
      double sum = 0.0;
      for(int j = 0; j < 2; ++j) sum += n.w1[i][j] * in[j];
      hidden[i] = sigmoid(sum);
   }
   double o = 0.0;
   for(int i = 0; i < 2; ++i) o += n.w2[i] * hidden[i];
   out = sigmoid(o);
}

//--- backpropagation do Pine. ATENCAO aos dois desvios preservados:
//    (a) no modo simples nao ha derivada da sigmoid;
//    (b) nos dois modos, w1 usa o w2 JA atualizado nesta iteracao.
void backprop(const Net &n, const double in[2], double actual, double predicted,
              const double hidden[2], double lr, bool simple)
{
   const double d_loss_d_output = 2.0 * (predicted - actual);

   for(int i = 0; i < 2; ++i)
      n.w2[i] -= lr * d_loss_d_output * hidden[i];

   for(int i = 0; i < 2; ++i)
      for(int j = 0; j < 2; ++j)
      {
         if(simple)
            n.w1[i][j] -= lr * d_loss_d_output * n.w2[i] * in[j];
         else
         {
            const double dsig = hidden[i] * (1.0 - hidden[i]);
            n.w1[i][j] -= lr * d_loss_d_output * n.w2[i] * dsig * in[j];
         }
      }
}

//--- fillMatrixRandomly do Pine: seed base 1337, alterado por i + j*rows
void fill_random(double *w, int seed)
{
   for(int i = 0; i < 2; ++i)          // w1 (2x2)
      for(int j = 0; j < 2; ++j)
         w[i*2 + j] = lcg01((unsigned int)(seed + i + j * 2));

   for(int j = 0; j < 2; ++j)          // w2 (1x2)
      w[4 + j] = lcg01((unsigned int)(seed + 0 + j * 1));
}

} // namespace

//+------------------------------------------------------------------+
int __stdcall NnVersion(void) { return NN_ABI_VERSION; }

//+------------------------------------------------------------------+
int __stdcall NnInitWeights(double *weights, int count, int seed)
{
   if(weights == nullptr || count < NN_WEIGHTS) return -1;
   fill_random(weights, seed);
   return NN_WEIGHTS;
}

//+------------------------------------------------------------------+
int __stdcall NnTrain(
   const double *close, const double *high, int size, int atBar,
   double lr, int epochs, int simpleBackprop, int seed,
   double *weights, int weightsCount, int initWeights,
   double *lossCurve, int lossCapacity,
   double *predicted, double *actual, double *finalLoss, double *maxScaleOut)
{
   if(close == nullptr || high == nullptr || lossCurve == nullptr ||
      predicted == nullptr || actual == nullptr || finalLoss == nullptr ||
      weights == nullptr)                              return -1;
   if(weightsCount < NN_WEIGHTS)                       return -1;
   if(size <= 3 || atBar < 2 || atBar >= size)         return -1;
   if(lr < 0.00001)                                    return -1;   // Pine minval
   if(epochs < 10 || epochs > 1000)                    return -1;   // Pine 10..1000
   if(lossCapacity < 1)                                return -1;

   //--- max_scale: maior high ate a barra (Pine linhas 16-17)
   double maxScale = 0.0;
   for(int i = 0; i <= atBar; ++i) if(high[i] > maxScale) maxScale = high[i];
   if(maxScale <= 0.0) return -1;
   if(maxScaleOut != nullptr) *maxScaleOut = maxScale;

   if(initWeights != 0) fill_random(weights, seed);

   Net net(weights);

   const double in[2]   = { close[atBar-1] / maxScale, close[atBar-2] / maxScale };
   const double target  =   close[atBar]   / maxScale;

   int written = 0;
   double out = 0.0, hidden[2] = {0,0}, loss = 0.0;

   for(int e = 0; e < epochs; ++e)
   {
      feedforward(net, in, out, hidden);
      loss = mse(out, target);
      if(written < lossCapacity) lossCurve[written++] = loss;
      backprop(net, in, target, out, hidden, lr, simpleBackprop != 0);
   }

   //--- previsao final apos o treino (Pine linha 146)
   feedforward(net, in, out, hidden);

   *predicted = out * maxScale;      // standardize()
   *actual    = target * maxScale;
   //--- particularidade 3 do .h: o Pine exibe a perda da ultima EPOCA
   //    (array.get(loss_curve, epochs-1)), nao a perda pos-backprop.
   *finalLoss = loss;
   return written;
}
