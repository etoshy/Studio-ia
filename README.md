# Studio IA — Z-Image Turbo + Chatterbox no seu pod da RunPod

Servidor para ligar num pod, gerar tudo de uma vez e desligar. A API é aberta (sem chave) e trabalha com fila:
você manda os pedidos, acompanha e baixa os resultados.

- 🖼 **Imagens**: Z-Image Turbo, a partir de uma lista de prompts (JSON ou .txt com 1 por linha).
- 🗣️ **Voz**: Chatterbox, com três modelos:
  - **pt-BR dedicado** (Multilingual V3 Single Language Pack);
  - **Turbo**, em inglês, com tags `[laugh]` `[sigh]` `[chuckle]`…;
  - **Multilingual V3**, com 23 idiomas.
- 📦 **Perfis de voz** no mesmo .zip do 📦 Exportar/Importar do BestEditor.
- 🔗 O **BestEditor** (v84_121+) usa este pod direto: ⚙️ Configurações → 🟣 Runpod → 🖥️ Meu pod.
- 🏭 A aba **Produção no pod** do BestEditor (v84_122+) manda vários roteiros de uma vez: o pod narra, **transcreve com o
  Whisper** (tempo de cada palavra) e gera as imagens das cenas difíceis.

```
app/main.py               API (FastAPI) — não carrega modelo nenhum
workers/worker_images.py  gera imagens (ambiente "img")
workers/worker_tts.py     gera voz (ambiente "tts" — o Chatterbox fixa versões antigas de torch/diffusers)
common/store.py           fila em disco (/workspace/studio_data/jobs)
common/perfis.py          perfis de voz (/workspace/studio_data/perfis)
cliente/studio.py         cliente de linha de comando pro seu PC
```

---

## 1. Criar o pod (uma vez)

1. **GPU**:
   - **L40S 48 GB** é a recomendada: Z-Image e Chatterbox cabem juntos na placa.
   - **RTX 4090 24 GB** também funciona. Nela o Z-Image liga sozinho o *CPU offload*, então fica um pouco mais lento.
   - ⚠️ **Evite RTX 5090 e placas Blackwell**: o Chatterbox usa torch 2.6, que não roda nelas.
2. **Template**: *RunPod PyTorch* (Ubuntu + Python 3.11).
3. **Network Volume** de ~60 GB montado em `/workspace`. É onde ficam os modelos e os ambientes, para não baixar tudo de novo a cada sessão.
4. **Expose HTTP Ports**: `8000`.

## 2. Instalar (uma vez por volume)

No terminal web do pod:

```bash
cd /workspace
# envie a pasta studio-ia (pelo Jupyter, scp, ou git clone de um repo seu)
bash studio-ia/setup.sh        # ~10-15 min na primeira vez
```

## 3. Cada sessão

```bash
cd /workspace/studio-ia && git pull && bash setup.sh && bash start.sh
```

O `setup.sh` é rápido quando já está tudo instalado (só confere), e instala o que for novo depois de um `git pull`.

Ele mostra a URL: `https://<id-do-pod>-8000.proxy.runpod.net`.

- Os modelos carregam em segundo plano: uns 2-5 minutos na primeira vez, menos de 1 minuto depois.
- Acompanhe em `GET /status`, ou com `tail -f /workspace/studio_logs/*.log`.
- A documentação interativa fica em **`/docs`**. Dá para testar tudo pelo navegador.

Opções do `start.sh`:

| Variável | Efeito |
|---|---|
| `ENABLE_IMG=0` | só voz |
| `ENABLE_TTS=0` | só imagens |
| `API_KEY=segredo` | passa a exigir a chave (header `X-API-Key`) |
| `TTS_PRELOAD="pacote:pt-br,turbo"` | modelos de voz que já carregam ao ligar |
| `TTS_WORKERS=2` | duas narrações ao mesmo tempo na mesma placa (~1,5× mais rápido; precisa de ~8 GB de VRAM a mais — bom na L40S) |
| `WHISPER_MODELO=turbo` | modelo do Whisper da transcrição (`turbo` é rápido e bom; `large-v3` mais lento) |

No fim, baixe os resultados e **desligue o pod**. Use *Stop* para manter o volume; o que estiver no volume fica.

> 🔓 **Sem chave**: a URL do proxy é longa e aleatória, mas qualquer pessoa que tiver o link usa a sua GPU enquanto o pod
> está ligado. Não publique o link. Se quiser trancar, suba com `API_KEY=...` e coloque a mesma chave no site.

---

## 4. Usar pelo BestEditor

Em ⚙️ Configurações → 🟣 Runpod → **🖥️ Meu pod**:

1. Cole a URL do pod.
2. Escolha "Meu pod" em *De onde vêm as imagens*.
3. Marque **🗣️ Chatterbox e Chatterbox Turbo no pod**.

A partir daí, as imagens da fonte `runpod` e o Chatterbox saem do pod, sem mudar mais nada no site.

## 5. Usar pelo cliente (`cliente/studio.py`, só precisa de `pip install requests`)

```bash
set STUDIO_URL=https://xxxx-8000.proxy.runpod.net        # Windows (Linux/Mac: export)

python studio.py imagens prompts.txt --proporcao 16:9 --quantidade 2 --estilo-depois ", cinematic, 35mm"
python studio.py vozes-importar perfis_exportados_do_site.zip   # o .zip do 📦 Exportar do site
python studio.py vozes                                         # lista os perfis no pod
python studio.py voz ep01/ ep02/ ingles.zip                    # pastas ou zips; cada um vira um job
python studio.py vozes-exportar --saida perfis_pod.zip         # volta pro site em 📦 Importar perfis
python studio.py status
```

O cliente espera cada job terminar e baixa tudo para `resultados/<id-do-job>/`.

---

## API

### Imagens

`POST /imagens`:

```json
{
  "nome": "ep12-cenas",
  "proporcao": "16:9",
  "quantidade": 1,
  "passos": 9,
  "formato": "png",
  "estilo_antes": "",
  "estilo_depois": ", cinematic lighting, 35mm photo",
  "seed": null,
  "prompts": [
    "um farol no litoral ao pôr do sol",
    {"prompt": "thumbnail com o texto 'VOCÊ SABIA?'", "nome": "thumb", "quantidade": 4, "proporcao": "16:9-hd"}
  ]
}
```

- **Proporções**: `1:1` `16:9` `9:16` `4:3` `3:4` `3:2` `2:3` `21:9` `16:9-hd` (1920×1088) `9:16-hd`. Também aceita `largura`/`altura`.
- **Texto puro**: `POST /imagens/txt?proporcao=16:9`, com o corpo em `text/plain` e 1 prompt por linha.
- **Resultado**: os arquivos de imagem e um `metadados.json` com o prompt, a seed e o tempo de cada imagem.

### Voz por zip (vários de uma vez)

`POST /voz` com `multipart/form-data`. Envie um ou mais arquivos no campo `arquivos`; cada zip vira um job.

Opcionalmente, o campo `config` (JSON) sobrescreve o `config.json` de todos os zips.

Conteúdo de cada zip:

```
config.json      (opcional)
roteiro.txt      (um ou mais .txt; cada .txt vira um áudio)
voz.wav          (opcional: a voz a clonar; mp3/m4a/ogg/flac também servem)
```

Campos do `config.json` (todos opcionais):

| campo | padrão | |
|---|---|---|
| `idioma` | `pt-br` | `pt-br`, `pt-pt`, `es-mx`, `es-es`, `zh-cmn`, `hi` usam o pacote dedicado; `en` usa o Turbo; os outros 23 idiomas, o Multilingual V3 |
| `modelo` | `auto` | `auto`, `pacote`, `turbo` ou `multilingual` |
| `perfil` | — | nome de um perfil importado no pod (no lugar do `voz.wav`) |
| `exaggeration` | 0.5 | emoção (0.25 a 2). Mais alto deixa a fala mais dramática e mais rápida |
| `cfg_weight` | 0.5 | menor = fala mais calma (0.3 combina com `exaggeration` 0.7+) |
| `temperature` | 0.8 | |
| `pausa_ms` / `pausa_paragrafo_ms` | 250 / 700 | silêncio entre frases / entre parágrafos |
| `max_chars` | 250 | tamanho máximo de cada trecho mandado ao modelo |
| `formato` | `wav` | `wav`, `mp3` ou `ambos` |
| `legenda_srt` | true | gera um `.srt` com o tempo de cada trecho |
| `normalizar` | true | pico em -1 dBFS |
| `salvar_trechos` | false | também entrega cada trecho separado |
| `nome` | — | nome do arquivo final (quando há um .txt só) |

Marcações no roteiro (valem dali em diante):

| Marcação | Efeito |
|---|---|
| `{{emocao=0.8}}` | muda a emoção |
| `{{cfg=0.3}}` | muda o cfg_weight |
| `{{temperatura=0.7}}` | muda a temperatura |
| `{{pausa=300}}` | silêncio entre frases (ms) |
| `{{pausa_paragrafo=900}}` | silêncio entre parágrafos (ms) |
| `{{silencio=1500}}` | insere um silêncio uma única vez (ms) |
| `{{voz=outra.wav}}` | troca para outro áudio do zip |
| `{{perfil=maria}}` | troca para um perfil do pod (dá para fazer diálogos) |

Dá para juntar várias numa marcação só: `{{emocao=0.9, cfg=0.3}}`.

As tags `[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[cough]` `[groan]` `[sniff]` `[shush]` `[clear throat]` só funcionam no **Turbo (inglês)**.

Exemplos prontos em `exemplos/`.

### Voz por texto (é o que o site usa)

`POST /voz/texto` com `multipart/form-data`. Campos:

- `texto`;
- `motor`: `chatterbox` ou `chatterbox_turbo`;
- `idioma`;
- a voz: `perfil` **ou** o arquivo `referencia`;
- `exaggeration`, `cfg_weight`, `temperature`;
- `pacote_dedicado`;
- `marcacoes` (desligadas por padrão);
- `transcrever`: depois de narrar, transcreve com o Whisper e entrega `<nome>.transcricao.json` (mesmo formato do
  `whisper_transcribe.py` do site: trechos com o tempo de cada palavra).

### Perfis de voz

| Rota | O que faz |
|---|---|
| `GET /vozes` | lista os perfis |
| `POST /vozes/importar` | recebe o `arquivo` (.zip). Aceita o export do site (com `manifesto.json`) ou um zip só com áudios. Opções: `nome`, `substituir`, `dividir` |
| `POST /vozes/{nome}/audios` | cria um perfil com áudios avulsos (um áudio longo é dividido em trechos de 10 s) |
| `GET /vozes/exportar?nomes=a,b` | devolve um .zip que o site importa em 📦 Importar perfis |
| `DELETE /vozes/{nome}` | apaga um perfil |

### Jobs

| Rota | O que faz |
|---|---|
| `GET /jobs?tipo=imagem\|voz&status=...` | lista os jobs |
| `GET /jobs/{id}` | status, progresso, posição na fila e links |
| `GET /jobs/{id}/zip` | baixa tudo do job · `GET /jobs/{id}/arquivos/{nome}` baixa um arquivo |
| `POST /lote/zip` `{"ids": [...]}` | vários jobs num zip só |
| `POST /jobs/{id}/cancelar` · `DELETE /jobs/{id}` | cancela · apaga |
| `GET /status` | estado dos workers (carregando/pronto/ocupado), GPU e contagem da fila |

---

## Testar no PC sem GPU (modo simulado)

O modo simulado gera imagens coloridas e um "bip" no lugar da voz. Serve para testar o fluxo e o cliente sem gastar nada.

```bash
pip install -r requirements/local-teste.txt
export MOCK=1 DATA_DIR=./dados
python workers/worker_images.py &  python workers/worker_tts.py &
python -m uvicorn app.main:app --port 8000
```

## Problemas comuns

- **`/status` mostra um worker `offline`**: veja `/workspace/studio_logs/imagens.log` ou `voz.log`. O erro mais comum é falta de VRAM numa placa de 24 GB; use `IMG_CPU_OFFLOAD=1`.
- **Sotaque estranho no pt-BR**: suba com `TTS_PACOTE_S3GEN=base`. Isso usa o decoder padrão no lugar do que vem no pacote.
- **Áudio de referência**: 10 a 20 s, uma pessoa só, sem música. Só os primeiros ~10 s entram na clonagem.
- **Texto muito curto** (uma palavra) às vezes falha no Chatterbox. O servidor junta frases curtas automaticamente.
