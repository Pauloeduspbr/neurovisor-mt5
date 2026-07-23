//+------------------------------------------------------------------+
//|                                       TV_04_NeuralNet.mq5        |
//|   Porte MQL5 do "Neural Network Showcase"                        |
//|   Pine v5 original: (c) Alien_Algorithms                         |
//|                                                                  |
//|   LICENCA: CC BY-NC-SA 4.0 (NonCommercial + ShareAlike).         |
//|   Uso pessoal/estudo livre; uso COMERCIAL nao permitido.         |
//|   https://creativecommons.org/licenses/by-nc-sa/4.0/            |
//|                                                                  |
//| Rede 2-2-1 (sigmoid, perda MSE) treinada em neural_net.dll (C++).|
//| Como no original, o treino roda SO na ultima barra confirmada e  |
//| todo o resultado e desenhado com objetos graficos.               |
//|                                                                  |
//| NOTA: os PESOS INICIAIS nao sao identicos aos do TradingView —    |
//| o math.random() do Pine nao e publico. A rede converge do mesmo  |
//| jeito, mas os numeros exatos diferem. Detalhes no .h da DLL.     |
//|                                                                  |
//| Requer "Permitir importacoes de DLL" HABILITADO.                 |
//+------------------------------------------------------------------+
#property copyright "Porte C++/MQL5 — original (c) Alien_Algorithms (CC BY-NC-SA 4.0)"
#property link      "https://creativecommons.org/licenses/by-nc-sa/4.0/"
#property version   "1.00"
#property description "Neural Network Showcase — rede 2-2-1, engine C++ via neural_net.dll"
#property description "Original CC BY-NC-SA 4.0 — uso comercial NAO permitido."

#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

#define EXPECTED_ABI_VERSION 2
#define OBJ_PREFIX "TV04_NN_"
#define MAX_EPOCHS 1000
#define NN_WEIGHTS 6

#import "neural_net.dll"
int NnVersion(void);
int NnInitWeights(double &weights[], int count, int seed);
int NnTrain(const double &close[], const double &high[], int size, int atBar,
            double lr, int epochs, int simpleBackprop, int seed,
            double &weights[], int weightsCount, int initWeights,
            double &lossCurve[], int lossCapacity,
            double &predicted, double &actual, double &finalLoss, double &maxScale);
#import

//--- ATENCAO: nada de `input group` (desloca os parametros de iCustom)
input double InpLR         = 0.1;    // Learning Rate
input int    InpEpochs     = 60;     // Epochs (10..1000)
input bool   InpSimpleBP   = false;  // Simple Backpropagation (versao didatica, incorreta)
input int    InpSeed       = 1337;   // Semente dos pesos iniciais
input bool   InpAccumulate = false;  // Acumular treino entre barras (NAO fiel ao Pine)
input bool   InpShowPips   = true;   // Mostrar o MSE tambem convertido em pips
input bool   InpPlotLoss   = true;   // Plot Loss Curve
input int    InpScaling    = 1;      // Chart Scaling Factor
input int    InpHOffset    = 100;    // Chart Horizontal Offset
input int    InpVOffset    = -2;     // Chart Vertical Offset (%)
input int    InpProjBars   = 40;     // Barras de projecao das linhas (Pine: 40)

//--- Os pesos vivem AQUI, por instancia do indicador — nao dentro da DLL.
//    Ver a nota de ABI 2 em cpp/04_neural_net/neural_net.h.
double   g_weights[NN_WEIGHTS];
datetime g_lastRun = 0;
double   g_pip     = 0.0;

//+------------------------------------------------------------------+
int OnInit()
{
   const int v = NnVersion();
   if(v != EXPECTED_ABI_VERSION)
   {
      PrintFormat("[TV_04_NN] ABI incompativel: neural_net.dll=%d, esperado=%d. "
                  "Recompile a DLL (cpp/04_neural_net/build.sh).", v, EXPECTED_ABI_VERSION);
      return INIT_FAILED;
   }
   if(InpEpochs < 10 || InpEpochs > MAX_EPOCHS)
   { PrintFormat("[TV_04_NN] Epochs fora de [10,%d]: %d", MAX_EPOCHS, InpEpochs); return INIT_PARAMETERS_INCORRECT; }
   if(InpLR < 0.00001)
   { Print("[TV_04_NN] Learning Rate muito baixo (minval 0.00001)."); return INIT_PARAMETERS_INCORRECT; }
   if(InpScaling < 1)
   { Print("[TV_04_NN] Chart Scaling Factor deve ser >= 1."); return INIT_PARAMETERS_INCORRECT; }

   g_pip = (_Digits == 3 || _Digits == 5) ? _Point*10 : _Point;
   g_lastRun = 0;

   //--- so importa no modo acumulativo; no modo fiel a DLL re-sorteia
   //    a cada barra e este estado inicial e descartado.
   if(NnInitWeights(g_weights, NN_WEIGHTS, InpSeed) != NN_WEIGHTS)
   { Print("[TV_04_NN] NnInitWeights falhou."); return INIT_FAILED; }

   if(InpAccumulate)
      Print("[TV_04_NN] Modo ACUMULATIVO ligado: o treino continua a cada barra. "
            "Isso NAO e o comportamento do Pine (que treina uma unica vez) e faz a "
            "rede convergir ate Predicted colar em Actual — medido: ~20 barras.");

   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("NN 2-2-1 (lr=%.4f, %d epochs)", InpLR, InpEpochs));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, OBJ_PREFIX);
   ChartRedraw();
}

//+------------------------------------------------------------------+
void HLine(const string n, const datetime t1, const datetime t2,
           const double p, const color c, const int w, const int style)
{
   if(ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_TREND, 0, t1, p, t2, p);
      ObjectSetInteger(0, n, OBJPROP_RAY_RIGHT, false);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
   }
   ObjectSetInteger(0, n, OBJPROP_COLOR, c);
   ObjectSetInteger(0, n, OBJPROP_WIDTH, w);
   ObjectSetInteger(0, n, OBJPROP_STYLE, style);
   ObjectMove(0, n, 0, t1, p);
   ObjectMove(0, n, 1, t2, p);
}

void Txt(const string n, const datetime t, const double p,
         const string s, const color c)
{
   if(ObjectFind(0, n) < 0)
   {
      ObjectCreate(0, n, OBJ_TEXT, 0, t, p);
      ObjectSetInteger(0, n, OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, n, OBJPROP_ANCHOR, ANCHOR_LEFT);
      ObjectSetInteger(0, n, OBJPROP_SELECTABLE, false);
   }
   ObjectSetInteger(0, n, OBJPROP_COLOR, c);
   ObjectSetString(0, n, OBJPROP_TEXT, s);
   ObjectMove(0, n, 0, t, p);
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < 10) return 0;

   //--- Pine: barstate.islastconfirmedhistory — treina uma vez por barra
   const int last = rates_total - 2;      // ultima barra CONFIRMADA
   if(last < 3) return rates_total;
   if(g_lastRun == time[last]) return rates_total;
   g_lastRun = time[last];

   double curve[];
   ArrayResize(curve, InpEpochs);
   ArraySetAsSeries(curve, false);
   double pred = 0.0, act = 0.0, fin = 0.0, scale = 0.0;

   //--- initWeights = 1 (default) reproduz o Pine: cada barra e uma sessao
   //    de treino independente, partindo de pesos recem-sorteados.
   const int n = NnTrain(close, high, rates_total, last,
                         InpLR, InpEpochs, InpSimpleBP ? 1 : 0, InpSeed,
                         g_weights, NN_WEIGHTS, InpAccumulate ? 0 : 1,
                         curve, InpEpochs, pred, act, fin, scale);
   if(n <= 0)
   {
      PrintFormat("[TV_04_NN] NnTrain falhou (rc=%d, bar=%d)", n, last);
      return rates_total;
   }

   ObjectsDeleteAll(0, OBJ_PREFIX);

   const int      sec  = PeriodSeconds(_Period);
   const datetime t0   = time[last];
   const datetime tEnd = t0 + (datetime)sec * InpProjBars;

   //--- Pine 153-157: linha roxa = previsto, verde = alvo
   HLine(OBJ_PREFIX+"PRED", t0, tEnd, pred, clrMediumPurple, 4, STYLE_SOLID);
   Txt  (OBJ_PREFIX+"PREDT", tEnd, pred, "Predicted Output", clrMediumPurple);
   HLine(OBJ_PREFIX+"ACT",  t0, tEnd, act,  clrLimeGreen,    4, STYLE_SOLID);
   Txt  (OBJ_PREFIX+"ACTT", tEnd, act,  "Actual Output",  clrLimeGreen);

   //--- Pine 159-163: conector tracejado + rotulo da perda
   {
      const datetime tm = t0 + (datetime)sec * 10;
      const string ln = OBJ_PREFIX+"CONN";
      if(ObjectFind(0, ln) < 0)
      {
         ObjectCreate(0, ln, OBJ_TREND, 0, tm, pred, tm, act);
         ObjectSetInteger(0, ln, OBJPROP_RAY_RIGHT, false);
         ObjectSetInteger(0, ln, OBJPROP_SELECTABLE, false);
      }
      ObjectSetInteger(0, ln, OBJPROP_COLOR, C'177,177,177');
      ObjectSetInteger(0, ln, OBJPROP_STYLE, STYLE_DASH);
      ObjectMove(0, ln, 0, tm, pred);
      ObjectMove(0, ln, 1, tm, act);

      //--- O MSE e medido no espaco NORMALIZADO e e o erro AO QUADRADO,
      //    entao um numero de aparencia minuscula pode valer dezenas de
      //    pips: erro_em_preco = sqrt(loss) * maxScale. Sem essa conversao
      //    o rotulo passa a impressao de precisao que ele nao tem.
      string txt = StringFormat("MSE Loss: %.10f", fin);
      if(InpShowPips && g_pip > 0.0)
         txt += StringFormat("   (~%.1f pips)", MathSqrt(fin)*scale/g_pip);
      Txt(OBJ_PREFIX+"LOSS", tm, (pred+act)/2.0, txt, C'195,195,195');
   }

   //--- Pine 165-187: curva de perda desenhada ao lado do preco
   if(InpPlotLoss && n > 1)
   {
      double mx = curve[0], mn = curve[0];
      for(int i = 1; i < n; ++i)
      { if(curve[i] > mx) mx = curve[i]; if(curve[i] < mn) mn = curve[i]; }
      const double range = mx - mn;
      if(range > 0.0)
      {
         const double chartRange = (high[last]-low[last]) / InpScaling;
         const double scale      = chartRange / range;
         const double vpct       = 1.0 + (InpVOffset / 100.0);
         const double base       = close[last] * vpct;

         datetime pt = 0; double pv = 0.0;
         for(int i = 0; i < n; ++i)
         {
            const double norm = (curve[i]-mn)/range;
            const double y    = norm*scale + base;
            const datetime x  = t0 + (datetime)sec * (i + InpHOffset);
            if(i > 0)
            {
               const string sn = OBJ_PREFIX+"LC"+IntegerToString(i);
               ObjectCreate(0, sn, OBJ_TREND, 0, pt, pv, x, y);
               ObjectSetInteger(0, sn, OBJPROP_COLOR, C'194,208,0');
               ObjectSetInteger(0, sn, OBJPROP_RAY_RIGHT, false);
               ObjectSetInteger(0, sn, OBJPROP_SELECTABLE, false);
               ObjectSetInteger(0, sn, OBJPROP_BACK, true);
            }
            pt = x; pv = y;
         }
         Txt(OBJ_PREFIX+"LCT", pt, pv, "Loss Curve", C'194,208,0');
      }
   }

   ChartRedraw();
   return rates_total;
}
//+------------------------------------------------------------------+
