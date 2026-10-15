"""Perfis de voz guardados no pod — no MESMO formato de .zip do site (perfis_pacote.py do BestEditor).

Pasta: DATA_DIR/perfis/<nome>/ref_01.wav ... (+ _original/ e info.json)

Zip aceito na importação:
  1. o .zip exportado pelo site (📦 Exportar), com manifesto.json:
       manifesto.json  {"formato": "narration_site/perfis_de_voz", "versao": 1, "perfis": [{nome, info, arquivos, vozes_slug}]}
       perfis/<nome>/ref_01.wav ...   perfis/<nome>/_original/ref_01.wav
  2. um .zip comum só com áudios: áudios na raiz -> UM perfil; uma pasta por pessoa -> um perfil por pasta.
A exportação gera o formato 1, que o site importa de volta em 🗣️ Perfis de voz -> 📦 Importar perfis.
"""
from __future__ import annotations

import io
import json
import posixpath
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from . import store

FORMATO = "narration_site/perfis_de_voz"
VERSAO = 1
PASTA_ORIGINAL = "_original"
MAX_CLIPES = 5
EXT_AUDIO = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm", ".mp4", ".mkv", ".mov", ".wma")
MAX_ARQUIVO = 200 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024


class PerfilErro(Exception):
    pass


def pasta_base() -> Path:
    p = store.DATA_DIR / "perfis"
    p.mkdir(parents=True, exist_ok=True)
    return p


def nome_seguro(nome: str) -> str:
    nome = (nome or "").strip()
    seguro = "".join(c for c in nome if c.isalnum() or c in ("_", "-", " ")).strip().replace(" ", "_")
    if not seguro:
        raise PerfilErro(f"nome de perfil inválido: {nome!r}")
    return seguro[:80]


def _pasta(nome: str) -> Path:
    return pasta_base() / nome_seguro(nome)


def listar() -> list[dict]:
    out = []
    for d in sorted(pasta_base().iterdir()):
        if d.is_dir():
            clipes = sorted(p.name for p in d.glob("*.wav"))
            if clipes:
                out.append({"nome": d.name, "clipes": clipes, "info": store.read_json(d / "info.json", {}) or {}})
    return out


def existe(nome: str) -> bool:
    try:
        return any(_pasta(nome).glob("*.wav"))
    except PerfilErro:
        return False


def clipes(nome: str) -> list[Path]:
    arquivos = sorted(_pasta(nome).glob("*.wav"))
    if not arquivos:
        raise PerfilErro(f"perfil de voz '{nome}' não existe no pod. Perfis: {', '.join(p['nome'] for p in listar()) or '(nenhum)'}")
    return arquivos


def excluir(nome: str) -> bool:
    d = _pasta(nome)
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False


# ------------------------------------------------------------ áudio


def _ffmpeg(*args: str) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, timeout=180)
    return r.returncode == 0


def _duracao(caminho: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(caminho)],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _processar_audios(origens: list[Path], destino: Path, dividir: bool = True) -> int:
    """Igual ao 'Novo perfil' do site: wav mono 22 kHz, até 25 s por clipe, até 5 clipes.
    Um único áudio longo é dividido em trechos de 10 s espalhados por ele (começo, meio, fim)."""
    destino.mkdir(parents=True, exist_ok=True)
    trechos: list[tuple[Path, float, float]] = []
    if len(origens) == 1 and dividir:
        dur = _duracao(origens[0])
        if dur > 30:
            n = min(MAX_CLIPES, int(dur // 10))
            passo = (dur - 10) / max(1, n - 1)
            trechos = [(origens[0], i * passo, 10.0) for i in range(n)]
    if not trechos:
        trechos = [(o, 0.0, 25.0) for o in origens[:MAX_CLIPES]]
    feitos = 0
    for o, inicio, dur in trechos:
        alvo = destino / f"ref_{feitos + 1:02d}.wav"
        if _ffmpeg("-ss", f"{inicio:.2f}", "-i", str(o), "-t", str(dur), "-ac", "1", "-ar", "22050",
                   "-c:a", "pcm_s16le", str(alvo)):
            feitos += 1
    if not feitos:
        raise PerfilErro("não consegui converter nenhum áudio (o ffmpeg está instalado? o arquivo é um áudio válido?)")
    return feitos


# ------------------------------------------------------------ importar


def _caminho_seguro(n: str) -> str | None:
    n = n.replace("\\", "/")
    if n.startswith("/") or (len(n) > 1 and n[1] == ":"):
        return None
    n = posixpath.normpath(n)
    partes = n.split("/")
    if n in (".", "") or ".." in partes or "__MACOSX" in partes or any(p.startswith("._") for p in partes) \
            or partes[-1] in (".DS_Store", "Thumbs.db", "desktop.ini"):
        return None
    return n


def _nome_livre(base: str, existentes: set) -> str:
    if base not in existentes:
        return base
    k = 2
    while f"{base}_{k}" in existentes:
        k += 1
    return f"{base}_{k}"


def _destino(nome: str, substituir: bool, existentes: set) -> str:
    if nome in existentes:
        if substituir:
            excluir(nome)
        else:
            nome = _nome_livre(nome, existentes)
    return nome


def importar(zip_path: Path, nome: str = "", substituir: bool = False, dividir: bool = True,
             nome_zip: str = "") -> list[dict]:
    try:
        z = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PerfilErro("esse arquivo não é um .zip válido") from exc
    with z:
        total = 0
        membros: dict[str, str] = {}
        for i in z.infolist():
            if i.is_dir():
                continue
            if i.file_size > MAX_ARQUIVO:
                raise PerfilErro(f"arquivo grande demais dentro do zip: {i.filename}")
            total += i.file_size
            seguro = _caminho_seguro(i.filename)
            if seguro:
                membros[i.filename] = seguro
        if total > MAX_TOTAL:
            raise PerfilErro("o zip descompactado passa de 1 GB")
        existentes = {p["nome"] for p in listar()}
        manifesto = None
        man = next((s for s, r in membros.items() if r == "manifesto.json"), None)
        if man:
            try:
                manifesto = json.loads(z.read(man).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                manifesto = None
        if isinstance(manifesto, dict) and manifesto.get("formato") == FORMATO:
            res = _importar_manifesto(z, membros, manifesto, substituir, (nome or "").strip() or None, existentes)
        else:
            padrao = (nome or "").strip() or Path(nome_zip or zip_path.name).stem or "perfil_importado"
            res = _importar_soltos(z, membros, padrao, substituir, dividir, existentes)
    if not res:
        raise PerfilErro("nenhum perfil importado: o zip não tinha clipes de áudio")
    return res


def _importar_manifesto(z, membros, manifesto, substituir, renomear, existentes) -> list[dict]:
    perfis = [p for p in (manifesto.get("perfis") or []) if isinstance(p, dict) and p.get("nome")]
    resultado = []
    for p in perfis:
        original = nome_seguro(str(p["nome"]))
        desejado = nome_seguro(renomear) if (renomear and len(perfis) == 1) else original
        prefixo = f"perfis/{p['nome']}/"
        arquivos = [(seg, membros[seg][len(prefixo):]) for seg in membros if membros[seg].startswith(prefixo)]
        if not any("/" not in rel and rel.lower().endswith(".wav") for _, rel in arquivos):
            continue
        nome_final = _destino(desejado, substituir, existentes)
        pasta = _pasta(nome_final)
        pasta.mkdir(parents=True, exist_ok=True)
        n = 0
        for seg, rel in arquivos:
            partes = rel.split("/")
            if not rel.lower().endswith(".wav") or len(partes) > 2 or (len(partes) == 2 and partes[0] != PASTA_ORIGINAL):
                continue
            alvo = pasta.joinpath(*partes)
            alvo.parent.mkdir(parents=True, exist_ok=True)
            with z.open(seg) as src, open(alvo, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += len(partes) == 1
        info = p.get("info") if isinstance(p.get("info"), dict) else {}
        store.write_json(pasta / "info.json", dict(info, vozes_slug=p.get("vozes_slug") or []))
        existentes.add(nome_final)
        resultado.append({"nome": nome_final, "de": str(p["nome"]), "clipes": n, "renomeado": nome_final != desejado})
    return resultado


def _importar_soltos(z, membros, padrao, substituir, dividir, existentes) -> list[dict]:
    grupos: dict[str, list[str]] = {}
    for seg in sorted(membros, key=lambda s: membros[s]):
        rel = membros[seg]
        if rel.lower().endswith(EXT_AUDIO):
            grupos.setdefault("/".join(rel.split("/")[:-1]), []).append(seg)
    if not grupos:
        raise PerfilErro("não achei nenhum áudio no zip (wav, mp3, m4a, ogg, flac… ou vídeo)")
    chaves = list(grupos)
    if len(chaves) > 1 and len({c.split("/")[0] for c in chaves}) == 1 and all("/" in c for c in chaves):
        grupos = {c.split("/", 1)[1]: v for c, v in grupos.items()}
    resultado = []
    with tempfile.TemporaryDirectory(prefix="perfil_") as tmp:
        for k, (grupo, segs) in enumerate(sorted(grupos.items())):
            base = (padrao if (len(grupos) == 1 or not grupo) else posixpath.basename(grupo)) or padrao
            nome_final = _destino(nome_seguro(base), substituir, existentes)
            origens = []
            for j, seg in enumerate(segs[:MAX_CLIPES]):
                alvo = Path(tmp) / f"g{k}_{j}{Path(membros[seg]).suffix.lower()}"
                with z.open(seg) as src, open(alvo, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                origens.append(alvo)
            n = _processar_audios(origens, _pasta(nome_final), dividir)
            existentes.add(nome_final)
            resultado.append({"nome": nome_final, "de": grupo or "(raiz do zip)", "clipes": n,
                              "renomeado": nome_final != nome_seguro(base)})
    return resultado


def importar_audios(nome: str, arquivos: list[Path], substituir: bool = True, dividir: bool = True) -> dict:
    """Cria um perfil direto de áudios enviados (sem zip)."""
    existentes = {p["nome"] for p in listar()}
    nome_final = _destino(nome_seguro(nome), substituir, existentes)
    n = _processar_audios(arquivos, _pasta(nome_final), dividir)
    return {"nome": nome_final, "clipes": n}


# ------------------------------------------------------------ exportar


def exportar(nomes: list[str] | None = None) -> bytes:
    todos = {p["nome"]: p for p in listar()}
    nomes = [nome_seguro(n) for n in (nomes or list(todos)) if n]
    if not nomes:
        raise PerfilErro("não há nenhum perfil de voz no pod")
    faltando = [n for n in nomes if n not in todos]
    if faltando:
        raise PerfilErro(f"perfil não encontrado: {', '.join(faltando)}")
    manifesto = {"formato": FORMATO, "versao": VERSAO, "criado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "perfis": []}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for nome in nomes:
            pasta = _pasta(nome)
            info = dict(todos[nome]["info"])
            vozes = info.pop("vozes_slug", [])
            arquivos = []
            for c in sorted(pasta.glob("*.wav")):
                z.write(c, f"perfis/{nome}/{c.name}")
                arquivos.append(c.name)
                o = pasta / PASTA_ORIGINAL / c.name
                if o.is_file():
                    z.write(o, f"perfis/{nome}/{PASTA_ORIGINAL}/{c.name}")
                    arquivos.append(f"{PASTA_ORIGINAL}/{c.name}")
            manifesto["perfis"].append({"nome": nome, "info": info, "arquivos": arquivos, "vozes_slug": vozes})
        z.writestr("manifesto.json", json.dumps(manifesto, ensure_ascii=False, indent=1))
        z.writestr("LEIA-ME.txt", "Perfis de voz exportados do pod (Studio IA).\n"
                                  "No site: 🗣️ Perfis de voz -> 📦 Importar perfis (.zip). Não precisa descompactar.\n")
    return buf.getvalue()
