#!/usr/bin/env bash
# Build da DLL x64 do Kalman D7 + deploy nos terminais MT5.
#   uso:  bash build.sh            -> compila e faz deploy
#         bash build.sh --no-copy  -> apenas compila
set -euo pipefail

# O g++ do MSYS2 carrega DLLs de mingw64/bin; sem esse dir no PATH o loader
# do Windows aborta antes do main (rc=1, stderr vazio).
MINGW_BIN="/c/msys64/mingw64/bin"
export PATH="$MINGW_BIN:$PATH"

GXX="$MINGW_BIN/g++.exe"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DLL="$SRC_DIR/neural_net.dll"

TERM_ROOT="/c/Users/MagnaTI/AppData/Roaming/MetaQuotes/Terminal"
PEPPERSTONE="$TERM_ROOT/73B7A2420D6397DFF9014A20F1201F97/MQL5/Libraries"   # alvo
TICKMILL="$TERM_ROOT/29E91DA909EB4475AB204481D1C2CE7D/MQL5/Libraries"      # host de teste

[ -x "$GXX" ] || { echo "ERRO: g++ nao encontrado em $GXX"; exit 1; }

echo "== compilando neural_net.dll (x64) =="
# ATENCAO: nada de -ffast-math aqui — o porte depende da propagacao
# correta de NaN para replicar o `na` do Pine.
"$GXX" -O2 -std=c++17 -Wall -Wextra -shared \
  -static -static-libgcc -static-libstdc++ \
  -fno-exceptions \
  -o "$OUT_DLL" \
  "$SRC_DIR/neural_net.cpp"

echo "== simbolos exportados =="
"$MINGW_BIN/objdump.exe" -p "$OUT_DLL" | grep -A6 "Ordinal/Name Pointer"

if [ "${1:-}" != "--no-copy" ]; then
  for d in "$PEPPERSTONE" "$TICKMILL"; do
    if [ -d "$d" ]; then
      cp -f "$OUT_DLL" "$d/" && echo "== deploy: $d/neural_net.dll"
    else
      echo "AVISO: diretorio nao encontrado, pulando: $d"
    fi
  done
fi

echo "OK"
