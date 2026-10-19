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

# Cria cada ambiente uma vez; o "pip install" roda sempre (rápido quando já está tudo lá) pra pegar
# dependências novas depois de um "git pull".
if [ ! -f "$VENVS/img/.ok" ]; then
  echo "==> criando o ambiente de imagens + API ($($PY_IMG --version))"
  rm -rf "$VENVS/img"
  criar_venv "$VENVS/img" "$PY_IMG"
  "$VENVS/img/bin/pip" install -q -U pip wheel
fi
# torch do tamanho do DRIVER deste pod: o torch mais novo do PyPI é feito pra CUDA 13 e não roda em driver 12.x
# ("The NVIDIA driver on your system is too old"). Lê a CUDA do driver no nvidia-smi e usa o índice do PyTorch certo.
TORCH_IDX=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_DRV="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' | head -1)"
  case "$CUDA_DRV" in
    13.*)        TORCH_IDX="https://download.pytorch.org/whl/cu130" ;;
    12.9|12.8)   TORCH_IDX="https://download.pytorch.org/whl/cu128" ;;
    12.7|12.6)   TORCH_IDX="https://download.pytorch.org/whl/cu126" ;;
    12.*)        TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
  esac
  echo "==> driver com CUDA ${CUDA_DRV:-?} → torch de ${TORCH_IDX:-PyPI}"
fi
torch_ok() { "$VENVS/img/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
if [ -n "$TORCH_IDX" ] && ! torch_ok; then
  echo "==> instalando o torch compatível com este driver"
  "$VENVS/img/bin/pip" install -q --force-reinstall torch --index-url "$TORCH_IDX"
fi
echo "==> dependências de imagens + API"
"$VENVS/img/bin/pip" install -q -r "$ROOT/requirements/img.txt" && touch "$VENVS/img/.ok"
if [ -n "$TORCH_IDX" ] && ! torch_ok; then
  echo "   aviso: o torch das imagens ainda não enxerga a GPU — veja: $VENVS/img/bin/python -c 'import torch; torch.cuda.init()'"
fi

# 🎮 a placa: as Blackwell (RTX 5090/5080, RTX PRO 6000, B200/B300) precisam de um torch novo no ambiente de voz
GPU=""
BLACKWELL=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "==> GPU: ${GPU:-?}"
  case "$GPU" in
    *5090*|*5080*|*5070*|*5060*|*Blackwell*|*B200*|*B300*|*GB200*|*"PRO 6000"*|*"Pro 6000"*) BLACKWELL=1 ;;
  esac
fi

if [ ! -f "$VENVS/tts/.ok" ]; then
  echo "==> criando o ambiente de voz ($($PY_TTS --version))"
  rm -rf "$VENVS/tts"
  criar_venv "$VENVS/tts" "$PY_TTS"
  "$VENVS/tts/bin/pip" install -q -U pip wheel
fi
if [ "$BLACKWELL" = 1 ] && [ -f "$VENVS/tts/.ok" ] && [ -f "$VENVS/tts/.torch_blackwell" ]; then
  # o Chatterbox pede torch==2.6: rodar o requirements de novo voltaria o torch velho (e a voz quebraria nesta placa)
  echo "==> dependências de voz: já instaladas (com o torch novo desta placa — pra reinstalar: bash setup.sh --refazer)"
else
  echo "==> dependências de voz (Chatterbox + Whisper)"
  rm -f "$VENVS/tts/.torch_blackwell"
  "$VENVS/tts/bin/pip" install -q -r "$ROOT/requirements/tts.txt" && touch "$VENVS/tts/.ok"
fi

voz_roda() {  # a voz usa LSTM: testa um de verdade na GPU (o torch 2.6 sem kernel da placa falha aqui)
  "$VENVS/tts/bin/python" -c "import torch; torch.nn.LSTM(4,4).cuda()(torch.zeros(1,1,4,device='cuda')); print('ok')" >/dev/null 2>&1
}
if [ "$BLACKWELL" = 1 ]; then
  if voz_roda; then
    echo "   voz: ok nesta placa"
    touch "$VENVS/tts/.torch_blackwell"
  else
    # o torch 2.6 do Chatterbox não tem os kernels da Blackwell ("no kernel image is available"): troca SÓ o torch do
    # ambiente de voz por um novo (o Chatterbox roda com ele). As outras placas continuam com o 2.6, como sempre.
    IDX_TTS="https://download.pytorch.org/whl/cu128"  # testado na RTX 5090 (torch 2.11 + torchaudio, Chatterbox narrando)
    echo "==> placa Blackwell: torch novo no ambiente de voz ($IDX_TTS)"
    "$VENVS/tts/bin/pip" install -q --upgrade torch torchaudio --index-url "$IDX_TTS" 2>&1 | grep -v "requires torch\|dependency resolver\|dependency conflicts" || true
    if voz_roda; then
      echo "   voz: ok nesta placa"
      touch "$VENVS/tts/.torch_blackwell"
    else
      echo "   aviso: a voz ainda não roda nesta placa — prefira RTX 3090/4090, L40S, A6000, A40, A100 ou H100."
    fi
  fi
elif [ -n "$GPU" ] && ! voz_roda; then
  echo "   aviso: o ambiente de voz não conseguiu usar a GPU — veja: $VENVS/tts/bin/python -c 'import torch; print(torch.cuda.is_available())'"
fi

echo
echo "Pronto. Agora rode:  bash $ROOT/start.sh"
