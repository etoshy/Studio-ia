#!/usr/bin/env bash
# Sobe a API (porta 8000) e os workers em segundo plano.
#   ENABLE_IMG=0 bash start.sh     -> só voz
#   ENABLE_TTS=0 bash start.sh     -> só imagens
#   API_KEY=segredo bash start.sh  -> exige chave (opcional)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="${WORKSPACE:-/workspace}"
VENVS="$WS/venvs"
export HF_HOME="${HF_HOME:-$WS/hf}"
export DATA_DIR="${DATA_DIR:-$WS/studio_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # menos memória "picada" na GPU
LOGS="${LOGS:-$WS/studio_logs}"
PORT="${PORT:-8000}"
mkdir -p "$LOGS" "$DATA_DIR" "$HF_HOME"

for v in img tts; do
  if [ ! -x "$VENVS/$v/bin/python" ]; then
    echo "Ambiente '$v' não encontrado. Rode antes: bash $ROOT/setup.sh"; exit 1
  fi
done

bash "$ROOT/stop.sh" >/dev/null 2>&1 || true
sleep 1

if [ "${ENABLE_IMG:-1}" = "1" ]; then
  nohup "$VENVS/img/bin/python" "$ROOT/workers/worker_images.py" >> "$LOGS/imagens.log" 2>&1 &
  echo "worker de imagens  pid $!  (log: $LOGS/imagens.log)"
fi
if [ "${ENABLE_TTS:-1}" = "1" ]; then
  # TTS_WORKERS=2 -> duas narrações ao mesmo tempo na mesma placa (precisa de ~8 GB de VRAM a mais)
  for n in $(seq 1 "${TTS_WORKERS:-1}"); do
    WORKER_ID=$n nohup "$VENVS/tts/bin/python" "$ROOT/workers/worker_tts.py" >> "$LOGS/voz-$n.log" 2>&1 &
    echo "worker de voz $n    pid $!  (log: $LOGS/voz-$n.log)"
  done
fi
cd "$ROOT"
nohup "$VENVS/img/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >> "$LOGS/api.log" 2>&1 &
echo "API                pid $!  (log: $LOGS/api.log)"

echo
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "URL da API:  https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
  echo "Documentação: https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/docs"
else
  echo "API em http://0.0.0.0:${PORT}  (docs em /docs)"
fi
echo "Os modelos carregam em segundo plano. Acompanhe com:  tail -f $LOGS/*.log"
echo "Ou veja GET /status até os dois workers aparecerem como 'pronto'."
