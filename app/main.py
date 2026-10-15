"""API do Studio IA: recebe pedidos, cria jobs na fila e entrega os resultados.

Não carrega nenhum modelo. Quem gera de fato são os workers:
    workers/worker_images.py  (Z-Image Turbo)
    workers/worker_tts.py     (Chatterbox)
"""
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Literal, Optional, Union

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import perfis, store  # noqa: E402

API_KEY = os.getenv("API_KEY", "").strip()
MAX_ZIP_MB = int(os.getenv("MAX_ZIP_MB", "500"))

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".webm"}

PROPORCOES = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "21:9": (1536, 640),
    "16:9-hd": (1920, 1088),
    "9:16-hd": (1088, 1920),
}


def autenticar(request: Request):
    """Sem chave por padrão. Se a variável API_KEY existir, exige header X-API-Key ou ?key=."""
    if not API_KEY:
        return
    chave = request.headers.get("x-api-key") or request.query_params.get("key")
    if chave != API_KEY:
        raise HTTPException(401, "API key inválida")


app = FastAPI(
    title="Studio IA – Imagens (Z-Image Turbo) e Voz (Chatterbox)",
    version="1.0.0",
    description="Envie listas de prompts ou zips de voz, acompanhe os jobs e baixe os resultados.",
    dependencies=[Depends(autenticar)],
)


@app.on_event("startup")
def _startup():
    store.ensure_dirs()


# ---------------------------------------------------------------- utilidades


def _job_ou_404(job_id: str) -> dict:
    try:
        st = store.get_status(job_id)
    except ValueError:
        raise HTTPException(400, "id inválido")
    if not st:
        raise HTTPException(404, "job não encontrado")
    return st


def _com_links(st: dict, request: Request | None = None) -> dict:
    jid = st["id"]
    out = dict(st)
    out["links"] = {
        "status": f"/jobs/{jid}",
        "zip": f"/jobs/{jid}/zip",
        "arquivos": [f"/jobs/{jid}/arquivos/{a}" for a in st.get("arquivos", [])],
    }
    if st["status"] == store.NA_FILA:
        fila = [s for s in store.list_jobs(st["tipo"]) if s["status"] in (store.NA_FILA, store.EM_ANDAMENTO)]
        out["posicao_fila"] = next((i for i, s in enumerate(fila) if s["id"] == jid), None)
    return out


def _dims(proporcao: Optional[str], largura: Optional[int], altura: Optional[int]) -> tuple[int, int]:
    if largura and altura:
        w, h = largura, altura
    else:
        if proporcao and proporcao not in PROPORCOES:
            raise HTTPException(400, f"proporcao inválida. Use uma de: {', '.join(PROPORCOES)}")
        w, h = PROPORCOES[proporcao or "16:9"]
    w, h = int(round(w / 16) * 16), int(round(h / 16) * 16)
    if not (256 <= w <= 2048 and 256 <= h <= 2048):
        raise HTTPException(400, "largura/altura devem ficar entre 256 e 2048")
    return w, h


# ---------------------------------------------------------------- imagens


class ItemImagem(BaseModel):
    prompt: str
    nome: Optional[str] = Field(None, description="nome do arquivo (sem extensão)")
    seed: Optional[int] = None
    proporcao: Optional[str] = None
    largura: Optional[int] = None
    altura: Optional[int] = None
    passos: Optional[int] = Field(None, ge=1, le=50)
    quantidade: Optional[int] = Field(None, ge=1, le=16)


class PedidoImagens(BaseModel):
    prompts: List[Union[str, ItemImagem]] = Field(..., min_length=1, max_length=5000)
    nome: Optional[str] = Field(None, description="nome do lote (aparece na listagem)")
    proporcao: str = "16:9"
    largura: Optional[int] = None
    altura: Optional[int] = None
    passos: int = Field(9, ge=1, le=50, description="Z-Image Turbo: 8-9 é o recomendado")
    seed: Optional[int] = Field(None, description="seed base; cada imagem usa seed+i. Vazio = aleatório")
    quantidade: int = Field(1, ge=1, le=16, description="variações por prompt")
    formato: Literal["png", "jpg", "webp"] = "png"
    estilo_antes: str = Field("", description="texto colado ANTES de cada prompt (estilo fixo do canal)")
    estilo_depois: str = Field("", description="texto colado DEPOIS de cada prompt")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nome": "ep12-cenas",
                "proporcao": "16:9",
                "estilo_depois": ", cinematic lighting, 35mm photo, highly detailed",
                "prompts": [
                    "um farol no litoral do Ceará ao pôr do sol",
                    {"prompt": "thumbnail com o texto 'VOCÊ SABIA?' em letras amarelas", "nome": "thumb", "quantidade": 4},
                ],
            }
        }
    }


def _criar_job_imagens(p: PedidoImagens) -> dict:
    itens, usados = [], set()
    base_seed = p.seed if p.seed is not None else random.randint(0, 2**31 - 1)
    n = 0
    for idx, raw in enumerate(p.prompts, start=1):
        it = ItemImagem(prompt=raw) if isinstance(raw, str) else raw
        texto = it.prompt.strip()
        if not texto:
            continue
        prompt_final = f"{p.estilo_antes}{texto}{p.estilo_depois}".strip()
        if it.proporcao or (it.largura and it.altura):
            w, h = _dims(it.proporcao, it.largura, it.altura)
        else:
            w, h = _dims(p.proporcao, p.largura, p.altura)
        qtd = it.quantidade or p.quantidade
        base_nome = store.slug(it.nome, 60) if it.nome else f"{idx:03d}_{store.slug(texto)}"
        for k in range(qtd):
            nome = base_nome if qtd == 1 else f"{base_nome}_v{k + 1}"
            while nome in usados:
                nome += "_b"
            usados.add(nome)
            seed = it.seed + k if it.seed is not None else base_seed + n
            itens.append({"nome": nome, "prompt": prompt_final, "prompt_original": texto, "seed": seed,
                          "largura": w, "altura": h, "passos": it.passos or p.passos})
            n += 1
    if not itens:
        raise HTTPException(400, "nenhum prompt válido")
    st = store.new_job("imagem", {"formato": p.formato, "itens": itens}, nome=p.nome)
    store.update_status(st["id"], progresso={"feito": 0, "total": len(itens)})
    return store.finalizar_criacao(st["id"])


@app.post("/imagens", tags=["imagens"], summary="Gerar uma lista de imagens")
def gerar_imagens(pedido: PedidoImagens):
    return _com_links(_criar_job_imagens(pedido))


@app.post("/imagens/txt", tags=["imagens"],
          summary="Gerar imagens a partir de texto puro (1 prompt por linha; linhas com # são ignoradas)")
def gerar_imagens_txt(
    texto: str = Body(..., media_type="text/plain"),
    nome: Optional[str] = Query(None),
    proporcao: str = Query("16:9"),
    passos: int = Query(9, ge=1, le=50),
    quantidade: int = Query(1, ge=1, le=16),
    formato: Literal["png", "jpg", "webp"] = Query("png"),
    seed: Optional[int] = Query(None),
    estilo_antes: str = Query(""),
    estilo_depois: str = Query(""),
):
    linhas = [l.strip() for l in texto.splitlines()]
    prompts = [l for l in linhas if l and not l.startswith("#")]
    if not prompts:
        raise HTTPException(400, "nenhum prompt encontrado no texto")
    pedido = PedidoImagens(prompts=prompts, nome=nome, proporcao=proporcao, passos=passos,
                           quantidade=quantidade, formato=formato, seed=seed,
                           estilo_antes=estilo_antes, estilo_depois=estilo_depois)
    return _com_links(_criar_job_imagens(pedido))


@app.get("/imagens/proporcoes", tags=["imagens"], summary="Proporções prontas")
def proporcoes():
    return {k: {"largura": w, "altura": h} for k, (w, h) in PROPORCOES.items()}


# ---------------------------------------------------------------- voz


def _extrair_zip(zpath: Path, destino: Path) -> None:
    limite = MAX_ZIP_MB * 1024 * 1024
    total = 0
    destino = destino.resolve()
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            partes = [p for p in info.filename.replace("\\", "/").split("/") if p not in ("", ".", "..")]
            if not partes or partes[0] == "__MACOSX" or partes[-1].startswith("._") or partes[-1] == ".DS_Store":
                continue
            total += info.file_size
            if total > limite:
                raise ValueError(f"conteúdo do zip passa de {MAX_ZIP_MB} MB")
            alvo = destino.joinpath(*partes).resolve()
            if destino not in alvo.parents:
                continue
            alvo.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(alvo, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _preparar_job_voz(upload: UploadFile, override: dict) -> dict:
    nome_arq = upload.filename or "pacote.zip"
    if not nome_arq.lower().endswith(".zip"):
        raise ValueError("envie um arquivo .zip")
    st = store.new_job("voz", {}, nome=Path(nome_arq).stem)
    jid = st["id"]
    d = store.job_dir(jid)
    try:
        zpath = d / "pacote.zip"
        with open(zpath, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        entrada = d / "input"
        try:
            _extrair_zip(zpath, entrada)
        except zipfile.BadZipFile:
            raise ValueError("zip inválido ou corrompido")
        zpath.unlink(missing_ok=True)

        arquivos = [p for p in entrada.rglob("*") if p.is_file()]
        roteiros = sorted(p for p in arquivos if p.suffix.lower() == ".txt")
        audios = sorted(p for p in arquivos if p.suffix.lower() in AUDIO_EXTS)
        configs = sorted((p for p in arquivos if p.name.lower() == "config.json"), key=lambda p: len(p.parts))
        if not roteiros:
            raise ValueError("o zip precisa ter pelo menos um .txt com o roteiro")

        config = {}
        if configs:
            try:
                config = json.loads(configs[0].read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as e:
                raise ValueError(f"config.json inválido: {e}")
            if not isinstance(config, dict):
                raise ValueError("config.json precisa ser um objeto { ... }")
        config.update(override)

        rel = lambda p: p.relative_to(entrada).as_posix()  # noqa: E731
        if config.get("perfil") and not perfis.existe(str(config["perfil"])):
            raise ValueError(f"perfil de voz '{config['perfil']}' não existe no pod (importe em POST /vozes/importar)")
        voz = config.get("voz")
        if voz is None and config.get("perfil"):
            voz = f"perfil:{config['perfil']}"
        if voz is None and audios:
            voz = rel(audios[0])
        params = {
            "config": config,
            "roteiros": [rel(p) for p in roteiros],
            "voz": voz,
            "audios": [rel(p) for p in audios],
        }
        store.write_json(d / "params.json", params)
        if config.get("nome"):
            store.update_status(jid, nome=str(config["nome"]))
        return store.finalizar_criacao(jid)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


@app.post("/voz", tags=["voz"], summary="Enviar 1 ou mais zips (voz + config.json + roteiro.txt)")
def gerar_voz(
    arquivos: List[UploadFile] = File(..., description="um ou vários .zip"),
    config: Optional[str] = Form(None, description='JSON que sobrescreve o config.json de todos os zips, ex: {"idioma":"en"}'),
):
    override = {}
    if config:
        try:
            override = json.loads(config)
            assert isinstance(override, dict)
        except Exception:
            raise HTTPException(400, "campo config precisa ser um JSON de objeto")
    jobs, erros = [], []
    for up in arquivos:
        try:
            jobs.append(_com_links(_preparar_job_voz(up, override)))
        except ValueError as e:
            erros.append({"arquivo": up.filename, "erro": str(e)})
    if not jobs and erros:
        raise HTTPException(400, {"erros": erros})
    return {"jobs": jobs, "erros": erros}


@app.post("/voz/texto", tags=["voz"], summary="Narrar um texto (usado pelo site): perfil do pod ou áudio de referência")
def gerar_voz_texto(
    texto: str = Form(..., description="o texto a narrar"),
    motor: Literal["auto", "chatterbox", "chatterbox_turbo"] = Form("auto",
        description="chatterbox = multilíngue (pt usa o pacote pt-br); chatterbox_turbo = inglês com tags"),
    idioma: str = Form("pt-br"),
    perfil: Optional[str] = Form(None, description="nome de um perfil de voz já importado no pod"),
    referencia: Optional[UploadFile] = File(None, description="áudio de referência pra clonar (alternativa ao perfil)"),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
    pacote_dedicado: bool = Form(True, description="usar o modelo dedicado do idioma (pt-br, es-mx…) quando existir"),
    marcacoes: bool = Form(False, description="interpretar {{emocao=0.8}} etc. no texto"),
    pausa_ms: int = Form(120),
    pausa_paragrafo_ms: int = Form(350),
    normalizar: bool = Form(False),
    legenda_srt: bool = Form(False),
    transcrever: bool = Form(False, description="transcreve com o Whisper depois de narrar (tempo de cada palavra)"),
    nome: Optional[str] = Form(None),
):
    if not texto.strip():
        raise HTTPException(400, "texto vazio")
    if perfil and not perfis.existe(perfil):
        raise HTTPException(404, f"perfil de voz '{perfil}' não existe no pod")
    if motor == "chatterbox_turbo":
        modelo = "turbo"
    elif motor == "chatterbox":
        modelo = "auto" if pacote_dedicado else "multilingual"
    else:
        modelo = "auto"
    if modelo == "auto" and motor == "chatterbox" and idioma.lower().startswith("en"):
        modelo = "multilingual"  # pediu o multilíngue em inglês: não troca pelo Turbo
    config = {"modelo": modelo, "idioma": idioma, "exaggeration": exaggeration, "cfg_weight": cfg_weight,
              "temperature": temperature, "marcacoes": marcacoes, "pausa_ms": pausa_ms,
              "pausa_paragrafo_ms": pausa_paragrafo_ms, "normalizar": normalizar, "legenda_srt": legenda_srt,
              "nome": nome or "narracao", "formato": "wav", "transcrever": transcrever}
    st = store.new_job("voz", {}, nome=nome or "narracao")
    d = store.job_dir(st["id"])
    (d / "input" / "roteiro.txt").write_text(texto, encoding="utf-8")
    voz, audios = None, []
    if referencia is not None and referencia.filename:
        ext = Path(referencia.filename).suffix.lower() or ".wav"
        if ext not in AUDIO_EXTS:
            ext = ".wav"
        alvo = d / "input" / f"referencia{ext}"
        with open(alvo, "wb") as f:
            shutil.copyfileobj(referencia.file, f)
        voz = alvo.name
        audios = [alvo.name]
    elif perfil:
        voz = f"perfil:{perfil}"
    store.write_json(d / "params.json", {"config": config, "roteiros": ["roteiro.txt"], "voz": voz, "audios": audios})
    return _com_links(store.finalizar_criacao(st["id"]))


# ---------------------------------------------------------------- perfis de voz


@app.get("/vozes", tags=["perfis de voz"], summary="Perfis de voz guardados no pod")
def listar_vozes():
    return perfis.listar()


@app.post("/vozes/importar", tags=["perfis de voz"],
          summary="Importar perfis: o .zip do 📦 Exportar do site, ou um zip só com áudios")
def importar_vozes(
    arquivo: UploadFile = File(...),
    nome: str = Form("", description="nome do perfil (zip comum com áudios na raiz, ou para renomear um perfil só)"),
    substituir: bool = Form(False, description="substitui perfil com o mesmo nome (senão vira nome_2)"),
    dividir: bool = Form(True, description="um áudio longo é dividido em trechos de 10 s"),
):
    store.ensure_dirs()
    tmp = store.DATA_DIR / f".import_{os.getpid()}_{random.randint(0, 1 << 30)}.zip"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(arquivo.file, f)
        return {"importados": perfis.importar(tmp, nome=nome, substituir=substituir, dividir=dividir,
                                              nome_zip=arquivo.filename or "")}
    except perfis.PerfilErro as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/vozes/{nome}/audios", tags=["perfis de voz"], summary="Criar/substituir um perfil com áudios avulsos")
def criar_voz(nome: str, arquivos: List[UploadFile] = File(...), dividir: bool = Form(True)):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        caminhos = []
        for i, up in enumerate(arquivos[: perfis.MAX_CLIPES]):
            alvo = Path(tmp) / f"a{i}{Path(up.filename or 'a.wav').suffix.lower()}"
            with open(alvo, "wb") as f:
                shutil.copyfileobj(up.file, f)
            caminhos.append(alvo)
        try:
            return perfis.importar_audios(nome, caminhos, substituir=True, dividir=dividir)
        except perfis.PerfilErro as e:
            raise HTTPException(400, str(e))


@app.get("/vozes/exportar", tags=["perfis de voz"],
         summary="Exportar perfis num .zip que o site importa (📦 Importar perfis)")
def exportar_vozes(nomes: Optional[str] = Query(None, description="separados por vírgula; vazio = todos")):
    lista = [n.strip() for n in (nomes or "").split(",") if n.strip()]
    try:
        dados = perfis.exportar(lista or None)
    except perfis.PerfilErro as e:
        raise HTTPException(400, str(e))
    store.ensure_dirs()
    destino = store.DATA_DIR / "perfis_export.zip"
    destino.write_bytes(dados)
    return FileResponse(destino, media_type="application/zip", filename="perfis_de_voz_pod.zip")


@app.delete("/vozes/{nome}", tags=["perfis de voz"], summary="Apagar um perfil do pod")
def apagar_voz(nome: str):
    try:
        if not perfis.excluir(nome):
            raise HTTPException(404, "perfil não encontrado")
    except perfis.PerfilErro as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/voz/idiomas", tags=["voz"], summary="Idiomas e modelos de voz disponíveis")
def idiomas():
    return {
        "modelos": {
            "auto": "escolhe sozinho pelo idioma (padrão)",
            "pacote": "modelo dedicado por idioma (pt-br, pt-pt, es-mx, es-es, zh-cmn, hi)",
            "turbo": "inglês, aceita tags [laugh] [chuckle] [sigh] [cough] [gasp] [groan] [sniff] [clear throat] [shush]",
            "multilingual": "Chatterbox Multilingual V3, 23 idiomas",
        },
        "auto": {
            "pt / pt-br": "pacote pt-br", "pt-pt": "pacote pt-pt", "es-mx / es-latam": "pacote es-mx-latam",
            "es-es": "pacote es-es", "zh-cmn": "pacote zh-cmn", "hi": "pacote hi", "en": "turbo (com tags)",
            "outros": "multilingual",
        },
        "multilingual": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
                         "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"],
    }


# ---------------------------------------------------------------- jobs


@app.get("/jobs", tags=["jobs"], summary="Listar jobs")
def listar_jobs(
    tipo: Optional[Literal["imagem", "voz"]] = None,
    status: Optional[str] = None,
    limite: int = Query(100, ge=1, le=1000),
):
    jobs = store.list_jobs(tipo)
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    jobs = list(reversed(jobs))[:limite]
    return [
        {k: j.get(k) for k in ("id", "tipo", "nome", "status", "criado", "concluido", "progresso", "mensagem")}
        for j in jobs
    ]


@app.get("/jobs/{job_id}", tags=["jobs"], summary="Status de um job")
def status_job(job_id: str):
    return _com_links(_job_ou_404(job_id))


def _montar_zip(destino: Path, pastas: list[tuple[str, Path]]) -> None:
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for prefixo, pasta in pastas:
            for f in sorted(pasta.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    arc = f.relative_to(pasta).as_posix()
                    z.write(f, f"{prefixo}/{arc}" if prefixo else arc)


@app.get("/jobs/{job_id}/zip", tags=["jobs"], summary="Baixar todos os resultados em .zip")
def baixar_zip(job_id: str):
    st = _job_ou_404(job_id)
    d = store.job_dir(job_id)
    destino = d / f"{job_id}.zip"
    _montar_zip(destino, [("", d / "output")])
    nome = f"{store.slug(st.get('nome') or '', 40) + '_' if st.get('nome') else ''}{job_id}.zip"
    return FileResponse(destino, media_type="application/zip", filename=nome)


@app.get("/jobs/{job_id}/arquivos/{nome:path}", tags=["jobs"], summary="Baixar um arquivo do resultado")
def baixar_arquivo(job_id: str, nome: str):
    _job_ou_404(job_id)
    base = (store.job_dir(job_id) / "output").resolve()
    alvo = (base / nome).resolve()
    if base not in alvo.parents or not alvo.is_file():
        raise HTTPException(404, "arquivo não encontrado")
    return FileResponse(alvo, filename=alvo.name)


@app.post("/jobs/{job_id}/cancelar", tags=["jobs"], summary="Cancelar (na fila ou em andamento)")
def cancelar(job_id: str):
    st = _job_ou_404(job_id)
    if st["status"] in store.FINAIS:
        return {"ok": False, "mensagem": f"job já está {st['status']}"}
    store.pedir_cancelamento(job_id)
    return {"ok": True, "mensagem": "cancelamento pedido; o que já foi gerado continua disponível"}


@app.delete("/jobs/{job_id}", tags=["jobs"], summary="Apagar job e arquivos")
def apagar(job_id: str):
    st = _job_ou_404(job_id)
    if st["status"] == store.EM_ANDAMENTO:
        raise HTTPException(409, "job em andamento; cancele antes")
    shutil.rmtree(store.job_dir(job_id), ignore_errors=True)
    return {"ok": True}


class PedidoLote(BaseModel):
    ids: List[str] = Field(..., min_length=1)


@app.post("/lote/zip", tags=["jobs"], summary="Baixar vários jobs num zip só (uma pasta por job)")
def baixar_lote(pedido: PedidoLote):
    pastas = []
    for jid in pedido.ids:
        st = _job_ou_404(jid)
        prefixo = f"{store.slug(st['nome'], 40)}_{jid}" if st.get("nome") else jid
        pastas.append((prefixo, store.job_dir(jid) / "output"))
    store.ensure_dirs()
    destino = store.DATA_DIR / "lote_temp.zip"
    _montar_zip(destino, pastas)
    return FileResponse(destino, media_type="application/zip", filename="lote.zip")


# ---------------------------------------------------------------- status geral


def _gpu() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return []
    gpus = []
    for linha in out.splitlines():
        nome, usado, total, util = [x.strip() for x in linha.split(",")]
        gpus.append({"nome": nome, "vram_usada_mb": int(usado), "vram_total_mb": int(total), "uso_pct": int(util)})
    return gpus


@app.get("/status", tags=["status"], summary="Workers, GPU e tamanho da fila")
def status_geral():
    jobs = store.list_jobs()
    contagem = {}
    for j in jobs:
        contagem.setdefault(j["tipo"], {}).setdefault(j["status"], 0)
        contagem[j["tipo"]][j["status"]] += 1
    return {"workers": store.workers_info(), "gpu": _gpu(), "jobs": contagem}


@app.get("/", tags=["status"], summary="Início")
def inicio():
    return {
        "app": "Studio IA",
        "documentacao_interativa": "/docs",
        "imagens": "POST /imagens (JSON) ou POST /imagens/txt (1 prompt por linha)",
        "voz": "POST /voz (multipart, campo 'arquivos' com 1 ou mais .zip)",
        "acompanhar": "GET /jobs/{id}",
        "baixar": "GET /jobs/{id}/zip",
        "status": "GET /status",
        "protegido_por_chave": bool(API_KEY),
    }
