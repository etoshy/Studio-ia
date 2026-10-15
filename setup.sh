#!/usr/bin/env bash
# Instala tudo no pod da RunPod. Rode uma vez por volume (os venvs ficam em /workspace).
# Uso:  bash setup.sh            (instala o que falta)
#       bash setup.sh --refazer  (apaga os venvs e reinstala)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="${WORKSPACE:-/workspace}"
VENVS="$WS/venvs"
export HF_HOME="${HF_HOME:-$WS/hf}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ "${1:-}" = "--refazer" ]; then
  rm -rf "$VENVS/img" "$VENVS/tts"
fi
mkdir -p "$VENVS" "$HF_HOME"

apt_instalar() {
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" >/dev/null
  fi
}

echo "==> ffmpeg"
command -v ffmpeg >/dev/null 2>&1 || apt_instalar ffmpeg || echo "   aviso: não consegui instalar ffmpeg (mp3 e referências .m4a não vão funcionar)"

criar_venv() {  # $1 = pasta  $2 = python
  "$2" -m venv "$1" 2>/dev/null || { apt_instalar python3-venv; "$2" -m venv "$1"; }
}

PY_IMG="${PYTHON:-python3}"
PY_TTS="${PYTHON_TTS:-}"
if [ -z "$PY_TTS" ]; then
  if command -v python3.11 >/dev/null 2>&1; then PY_TTS=python3.11; else PY_TTS=python3; fi
fi

if [ ! -f "$VENVS/img/.ok" ]; then
  echo "==> ambiente de imagens + API ($($PY_IMG --version))"
  rm -rf "$VENVS/img"
  criar_venv "$VENVS/img" "$PY_IMG"
  "$VENVS/img/bin/pip" install -q -U pip wheel
  "$VENVS/img/bin/pip" install -q -r "$ROOT/requirements/img.txt"
  touch "$VENVS/img/.ok"
else
  echo "==> ambiente de imagens já instalado"
fi

if [ ! -f "$VENVS/tts/.ok" ]; then
  echo "==> ambiente de voz ($($PY_TTS --version))"
  rm -rf "$VENVS/tts"
  criar_venv "$VENVS/tts" "$PY_TTS"
  "$VENVS/tts/bin/pip" install -q -U pip wheel
  "$VENVS/tts/bin/pip" install -q -r "$ROOT/requirements/tts.txt"
  touch "$VENVS/tts/.ok"
else
  echo "==> ambiente de voz já instalado"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "==> GPU: $GPU"
  case "$GPU" in
    *5090*|*5080*|*Blackwell*|*B200*|*"PRO 6000"*|*"Pro 6000"*)
      echo "   aviso: o Chatterbox usa torch 2.6, que não roda em placas Blackwell."
      echo "   Prefira RTX 4090, L40S, A6000, A100 ou H100." ;;
  esac
fi

echo
echo "Pronto. Agora rode:  bash $ROOT/start.sh"
