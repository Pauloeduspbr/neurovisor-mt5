//+------------------------------------------------------------------+
//| test_nn.cpp — teste de mesa da DLL neural_net.dll (ABI 2)        |
//|                                                                  |
//| O teste central (T4) e o unico que importa numa rede neural:     |
//| a PERDA TEM DE CAIR ao longo das epocas. Se nao cair, o          |
//| backpropagation esta errado, por mais que tudo compile.          |
//|                                                                  |
//| T9/T10 sao a regressao do defeito achado em 2026-07-22: a ABI 1  |
//| guardava os pesos em globais, o wrapper re-treinava a cada barra |
//| e a rede convergia ate Predicted colar no Actual.                |
//+------------------------------------------------------------------+
#include <windows.h>
#include <cstdio>
#include <cmath>
#include <vector>

typedef int (__stdcall *fn_ver_t)(void);
typedef int (__stdcall *fn_init_t)(double*, int, int);
typedef int (__stdcall *fn_train_t)(const double*, const double*, int, int,
                                    double, int, int, int,
                                    double*, int, int,
                                    double*, int, double*, double*, double*, double*);

static fn_ver_t   NnVersion = nullptr;
static fn_init_t  NnInitW   = nullptr;
static fn_train_t NnTrain   = nullptr;
static int g_fail = 0;

static void check(bool ok, const char *name, const char *detail = "")
{
   std::printf("%-6s %s %s\n", ok ? "[ OK ]" : "[FAIL]", name, detail);
   if(!ok) ++g_fail;
}

int main()
{
   HMODULE hm = LoadLibraryA("neural_net.dll");
   if(!hm){ std::printf("[FAIL] LoadLibrary erro %lu\n", GetLastError()); return 1; }
   NnVersion = (fn_ver_t)(void*)GetProcAddress(hm,"NnVersion");
   NnInitW   = (fn_init_t)(void*)GetProcAddress(hm,"NnInitWeights");
   NnTrain   = (fn_train_t)(void*)GetProcAddress(hm,"NnTrain");
   if(!NnVersion||!NnInitW||!NnTrain){ std::printf("[FAIL] GetProcAddress\n"); return 1; }

   check(NnVersion()==2, "T1 ABI version == 2");

   const int N = 500;
   std::vector<double> c(N), h(N);
   for(int i=0;i<N;++i)
   {
      c[i] = 100.0 + 5.0*std::sin(i/20.0) + 0.05*i;
      h[i] = c[i] + 0.5;
   }

   const int EP = 200;
   std::vector<double> curve(EP,0.0);
   double w[6] = {0,0,0,0,0,0};
   double pred=0, act=0, fin=0, scale=0;

   //--- T2 fail-fast
   {
      double cu[10], ww[6];
      const int r1 = NnTrain(nullptr,h.data(),N,N-1,0.1,60,0,1337,ww,6,1,cu,10,&pred,&act,&fin,&scale);
      const int r2 = NnTrain(c.data(),h.data(),N,N-1,0.1,5,0,1337,ww,6,1,cu,10,&pred,&act,&fin,&scale);   // epochs<10
      const int r3 = NnTrain(c.data(),h.data(),N,N-1,0.0,60,0,1337,ww,6,1,cu,10,&pred,&act,&fin,&scale);  // lr invalido
      const int r4 = NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,ww,3,1,cu,10,&pred,&act,&fin,&scale);  // pesos de menos
      check(r1==-1 && r2==-1 && r3==-1 && r4==-1, "T2 fail-fast em argumento invalido");
   }

   //--- T3 treino executa e devolve a curva completa
   {
      const int n = NnTrain(c.data(),h.data(),N,N-1,0.1,EP,0,1337,w,6,1,
                            curve.data(),EP,&pred,&act,&fin,&scale);
      char d[190];
      std::snprintf(d,sizeof(d),"(%d epocas, pred=%.5f actual=%.5f maxScale=%.5f)", n, pred, act, scale);
      check(n==EP && pred>0.0 && act>0.0 && scale>0.0, "T3 treino devolve curva e previsao", d);
   }

   //--- T4 CRITICO: a perda DIMINUI ao longo do treino
   {
      const double first = curve[0], last = curve[EP-1];
      double h1=0.0, h2=0.0;
      for(int i=0;i<EP/2;++i)      h1 += curve[i];
      for(int i=EP/2;i<EP;++i)     h2 += curve[i];
      h1 /= (EP/2); h2 /= (EP-EP/2);

      char d[192];
      std::snprintf(d,sizeof(d),"(loss %.6e -> %.6e ; media 1a metade=%.3e, 2a=%.3e)",
                    first, last, h1, h2);
      check(last < first && h2 < h1, "T4 a perda CAI durante o treino", d);
   }

   //--- T5 previsao fica proxima do alvo depois de treinar
   {
      const double err = std::fabs(pred-act)/act*100.0;
      char d[128]; std::snprintf(d,sizeof(d),"(erro=%.3f%% do preco)",err);
      check(err < 5.0, "T5 previsao proxima do alvo", d);
   }

   //--- T6 mais epocas => perda final menor ou igual
   {
      std::vector<double> c2(100,0.0), c3(400,0.0);
      double w2[6], w3[6], p2=0,a2=0,f2=0,s2=0, p3=0,a3=0,f3=0,s3=0;
      NnTrain(c.data(),h.data(),N,N-1,0.1,20, 0,1337,w2,6,1,c2.data(),100,&p2,&a2,&f2,&s2);
      NnTrain(c.data(),h.data(),N,N-1,0.1,400,0,1337,w3,6,1,c3.data(),400,&p3,&a3,&f3,&s3);
      char d[160];
      std::snprintf(d,sizeof(d),"(20 epocas=%.3e ; 400 epocas=%.3e)",f2,f3);
      check(f3 <= f2, "T6 treinar mais reduz a perda", d);
   }

   //--- T7 modo simples difere do verbose (o bug didatico do autor)
   {
      std::vector<double> cs(100,0.0), cv(100,0.0);
      double ws[6], wv[6];
      double ps=0,as_=0,fs=0,ss=0, pv=0,av=0,fv=0,sv=0;
      NnTrain(c.data(),h.data(),N,N-1,0.1,100,1,1337,ws,6,1,cs.data(),100,&ps,&as_,&fs,&ss);
      NnTrain(c.data(),h.data(),N,N-1,0.1,100,0,1337,wv,6,1,cv.data(),100,&pv,&av,&fv,&sv);
      char d[176];
      std::snprintf(d,sizeof(d),"(simples: loss=%.3e pred=%.5f | verbose: loss=%.3e pred=%.5f)",
                    fs,ps,fv,pv);
      check(std::fabs(ps-pv) > 1e-12, "T7 backprop simples != verbose", d);
   }

   //--- T8 os pesos saem treinados no array do CHAMADOR
   {
      double ww[6]; std::vector<double> cu(60,0.0);
      double p=0,a=0,f=0,s=0;
      NnInitW(ww,6,1337);
      const double w0 = ww[0], w4 = ww[4];
      NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,ww,6,0,cu.data(),60,&p,&a,&f,&s);
      char d[176];
      std::snprintf(d,sizeof(d),"(w[0] %.4f->%.4f  w[4] %.4f->%.4f)", w0,ww[0],w4,ww[4]);
      check(std::fabs(ww[0]-w0) > 1e-12 && std::fabs(ww[4]-w4) > 1e-12,
            "T8 pesos treinados voltam ao chamador", d);
   }

   //--- T9 REGRESSAO: a DLL NAO guarda estado entre chamadas.
   //    Duas chamadas identicas com initWeights=1 tem de dar o MESMO
   //    resultado. Na ABI 1 a segunda continuava o treino da primeira.
   {
      std::vector<double> cA(60,0.0), cB(60,0.0);
      double wA[6], wB[6], pA=0,aA=0,fA=0,sA=0, pB=0,aB=0,fB=0,sB=0;
      NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,wA,6,1,cA.data(),60,&pA,&aA,&fA,&sA);
      NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,wB,6,1,cB.data(),60,&pB,&aB,&fB,&sB);
      char d[176];
      std::snprintf(d,sizeof(d),"(1a: pred=%.6f loss=%.3e | 2a: pred=%.6f loss=%.3e)",
                    pA,fA,pB,fB);
      check(std::fabs(pA-pB) < 1e-15 && std::fabs(fA-fB) < 1e-18,
            "T9 chamadas repetidas sao IDENTICAS (sem estado)", d);
   }

   //--- T10 duas "instancias" (dois graficos) nao se contaminam:
   //    treinar a rede X entre dois treinos da rede Y nao muda Y.
   {
      std::vector<double> cu(60,0.0);
      std::vector<double> c2(N);
      for(int i=0;i<N;++i) c2[i] = 50.0 + 2.0*std::cos(i/9.0) + 0.02*i;

      double wY[6], wX[6], pY1=0,pY2=0,pX=0,a=0,f=0,s=0;
      NnTrain(c.data(), h.data(),N,N-1,0.1,60,0,1337,wY,6,1,cu.data(),60,&pY1,&a,&f,&s);
      NnTrain(c2.data(),h.data(),N,N-1,0.1,60,0,4242,wX,6,1,cu.data(),60,&pX, &a,&f,&s);
      NnTrain(c.data(), h.data(),N,N-1,0.1,60,0,1337,wY,6,1,cu.data(),60,&pY2,&a,&f,&s);
      char d[176];
      std::snprintf(d,sizeof(d),"(Y antes=%.6f  X no meio=%.6f  Y depois=%.6f)",pY1,pX,pY2);
      check(std::fabs(pY1-pY2) < 1e-15, "T10 instancias nao se contaminam", d);
   }

   //--- T11 modo acumulativo AINDA funciona quando pedido de proposito
   {
      std::vector<double> cu(60,0.0);
      double ww[6], p=0,a=0,f1=0,f2=0,s=0;
      NnInitW(ww,6,1337);
      NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,ww,6,0,cu.data(),60,&p,&a,&f1,&s);
      NnTrain(c.data(),h.data(),N,N-1,0.1,60,0,1337,ww,6,0,cu.data(),60,&p,&a,&f2,&s);
      char d[160];
      std::snprintf(d,sizeof(d),"(1a passada=%.3e -> 2a passada=%.3e)",f1,f2);
      check(f2 < f1, "T11 initWeights=0 continua o treino", d);
   }

   //--- T12 a conversao MSE->preco esta documentada e confere:
   //    |pred-act| deve bater com sqrt(loss)*maxScale a menos da
   //    defasagem de 1 epoca (particularidade 3 do .h).
   {
      std::vector<double> cu(400,0.0);
      double ww[6], p=0,a=0,f=0,s=0;
      NnTrain(c.data(),h.data(),N,N-1,0.1,400,0,1337,ww,6,1,cu.data(),400,&p,&a,&f,&s);
      const double gapReal = std::fabs(p-a);
      const double gapTeo  = std::sqrt(f)*s;
      const double rel     = std::fabs(gapReal-gapTeo)/gapTeo*100.0;
      char d[190];
      std::snprintf(d,sizeof(d),"(gap=%.6f  sqrt(loss)*scale=%.6f  dif=%.2f%%)",
                    gapReal,gapTeo,rel);
      check(rel < 15.0, "T12 sqrt(loss)*maxScale == gap de preco", d);
   }

   FreeLibrary(hm);
   std::printf("\n%s  (%d falha(s))\n", g_fail==0?"TODOS OS TESTES PASSARAM":"TESTES FALHARAM", g_fail);
   return g_fail==0?0:1;
}
