"""Fila de jobs baseada em disco, compartilhada entre a API e os workers.

Cada job vive em DATA_DIR/jobs/<id>/:
    status.json   estado atual (escrito pela API na criação e depois só pelo worker)
    params.json   parâmetros do pedido
    input/        arquivos enviados (zips extraídos etc.)
    output/       resultados
    cancelar      arquivo-sinal criado pela API para pedir cancelamento
"""
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/studio_data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
WORKERS_DIR = DATA_DIR / "workers"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Estados
PREPARANDO = "preparando"
NA_FILA = "na_fila"
EM_ANDAMENTO = "em_andamento"
CONCLUIDO = "concluido"
CONCLUIDO_COM_ERROS = "concluido_com_erros"
ERRO = "erro"
CANCELADO = "cancelado"
FINAIS = {CONCLUIDO, CONCLUIDO_COM_ERROS, ERRO, CANCELADO}


def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    WORKERS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            time.sleep(0.05)
    return default


def job_dir(job_id: str) -> Path:
    if not _ID_RE.match(job_id or ""):
        raise ValueError("id de job inválido")
    return JOBS_DIR / job_id


def new_job(tipo: str, params: dict, nome: str | None = None) -> dict:
    """Cria o job em estado 'preparando'. Chame finalizar_criacao() depois de gravar os inputs."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefixo = {"imagem": "img", "voz": "voz"}.get(tipo, tipo[:3])
    jid = f"{prefixo}-{ts}-{uuid.uuid4().hex[:6]}"
    d = JOBS_DIR / jid
    (d / "input").mkdir(parents=True)
    (d / "output").mkdir()
    st = {
        "id": jid,
        "tipo": tipo,
        "nome": nome,
        "status": PREPARANDO,
        "criado": agora(),
        "ordem": time.time_ns(),
        "iniciado": None,
        "concluido": None,
        "progresso": {"feito": 0, "total": 0},
        "arquivos": [],
        "erros": [],
        "avisos": [],
        "mensagem": "",
    }
    write_json(d / "params.json", params)
    write_json(d / "status.json", st)
    return st


def finalizar_criacao(job_id: str) -> dict:
    return update_status(job_id, status=NA_FILA)


def get_status(job_id: str) -> dict | None:
    d = job_dir(job_id)
    st = read_json(d / "status.json")
    if st and st["status"] == NA_FILA and (d / "cancelar").exists():
        st = dict(st, status=CANCELADO)
    return st


def update_status(job_id: str, **changes) -> dict:
    path = job_dir(job_id) / "status.json"
    st = read_json(path, {}) or {}
    st.update(changes)
    write_json(path, st)
    return st


def get_params(job_id: str) -> dict:
    return read_json(job_dir(job_id) / "params.json", {}) or {}


def list_jobs(tipo: str | None = None) -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    out = []
    for d in sorted(JOBS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not _ID_RE.match(d.name):
            continue
        st = get_status(d.name)
        if not st:
            continue
        if tipo and st.get("tipo") != tipo:
            continue
        out.append(st)
    out.sort(key=lambda s: (s.get("ordem") or 0, s.get("criado") or ""))
    return out


def pedir_cancelamento(job_id: str) -> None:
    (job_dir(job_id) / "cancelar").touch()


def cancel_requested(job_id: str) -> bool:
    return (job_dir(job_id) / "cancelar").exists()


class _Trava:
    """Trava entre processos (vários workers do mesmo tipo pegando jobs da mesma fila)."""

    def __init__(self, nome: str):
        ensure_dirs()
        self.f = open(WORKERS_DIR / f".{nome}.lock", "w")

    def __enter__(self):
        try:
            import fcntl

            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:  # Windows (modo de teste): sem trava
            pass
        return self

    def __exit__(self, *a):
        try:
            import fcntl

            fcntl.flock(self.f, fcntl.LOCK_UN)
        except ImportError:
            pass
        self.f.close()


def claim_next(tipo: str) -> dict | None:
    """Pega o job mais antigo na fila desse tipo e marca como em andamento (seguro com vários workers)."""
    with _Trava(f"fila_{tipo}"):
        return _claim_next(tipo)


def _claim_next(tipo: str) -> dict | None:
    for st in list_jobs(tipo):
        raw = read_json(job_dir(st["id"]) / "status.json") or {}
        if raw.get("status") != NA_FILA:
            continue
        if cancel_requested(st["id"]):
            update_status(st["id"], status=CANCELADO, concluido=agora(), mensagem="cancelado antes de iniciar")
            continue
        return update_status(st["id"], status=EM_ANDAMENTO, iniciado=agora(), mensagem="iniciando", worker=WORKER_ID)
    return None


WORKER_ID = os.getenv("WORKER_ID", "1")


def requeue_stale(tipo: str) -> None:
    """Se o worker reiniciou no meio de um job, devolve pra fila os jobs que ERAM DELE."""
    for st in list_jobs(tipo):
        raw = read_json(job_dir(st["id"]) / "status.json") or {}
        if raw.get("status") == EM_ANDAMENTO and str(raw.get("worker") or "1") == WORKER_ID:
            update_status(st["id"], status=NA_FILA, mensagem="reenfileirado após reinício do worker")


def heartbeat(tipo: str, estado: str, **extra) -> None:
    ensure_dirs()
    nome = tipo if WORKER_ID == "1" else f"{tipo}-{WORKER_ID}"
    write_json(WORKERS_DIR / f"{nome}.json", {"estado": estado, "ts": time.time(), "pid": os.getpid(), **extra})


def workers_info() -> dict:
    info = {}
    extras = sorted(p.stem for p in WORKERS_DIR.glob("*-*.json")) if WORKERS_DIR.exists() else []
    for tipo in ["imagem", "voz"] + extras:
        hb = read_json(WORKERS_DIR / f"{tipo}.json")
        if not hb:
            info[tipo] = {"estado": "nao_iniciado"}
            continue
        idade = time.time() - hb.get("ts", 0)
        if idade > 45:
            hb = dict(hb, estado="offline")
        hb["segundos_desde_sinal"] = round(idade, 1)
        info[tipo] = hb
    return info


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(texto: str, limite: int = 40) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    t = _SLUG_RE.sub("-", t).strip("-")
    return (t[:limite].rstrip("-")) or "item"
