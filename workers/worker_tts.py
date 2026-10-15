"""Worker de voz: pega jobs 'voz' da fila e narra os roteiros com Chatterbox.

Engines:
    turbo          Chatterbox-Turbo (inglês, aceita tags [laugh], [sigh] ...)
    multilingual   Chatterbox Multilingual V3 (23 idiomas)
    pacote:<loc>   Single Language Pack (pt-br, pt-pt, es-mx, es-es, zh-cmn, hi)
"""
import copy
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import perfis, store  # noqa: E402
from roteiro import montar_eventos  # noqa: E402

TIPO = "voz"
MOCK = os.getenv("MOCK") == "1"
DEVICE = os.getenv("TTS_DEVICE", "cuda")
MAX_MODELOS = int(os.getenv("TTS_MAX_MODELOS", "3"))
PRELOAD = [x.strip() for x in os.getenv("TTS_PRELOAD", "pacote:pt-br").split(",") if x.strip()]
MTL_VERSAO = os.getenv("TTS_MTL_VERSAO", "v3")
PACOTE_S3GEN = os.getenv("TTS_PACOTE_S3GEN", "v3")  # v3 = decoder que vem no pacote | base = s3gen.pt padrão
SR = 24000

PACOTES = {
    "pt-br": ("ResembleAI/Chatterbox-Multilingual-pt-br", "pt"),
    "pt-pt": ("ResembleAI/Chatterbox-Multilingual-pt-pt", "pt"),
    "es-mx": ("ResembleAI/Chatterbox-Multilingual-es-mx-latam", "es"),
    "es-es": ("ResembleAI/Chatterbox-Multilingual-es-es", "es"),
    "zh-cmn": ("ResembleAI/Chatterbox-Multilingual-zh-cmn", "zh"),
    "hi": ("ResembleAI/Chatterbox-Multilingual-hi", "hi"),
}
APELIDOS_PACOTE = {"pt": "pt-br", "pt-br": "pt-br", "ptbr": "pt-br", "pt-pt": "pt-pt", "es-mx": "es-mx",
                   "es-latam": "es-mx", "es-419": "es-mx", "es-es": "es-es", "zh-cmn": "zh-cmn", "hi": "hi"}
IDIOMAS_MTL = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
               "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}

PADRAO = {
    "modelo": "auto",
    "idioma": "pt-br",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "pausa_ms": 250,
    "pausa_paragrafo_ms": 700,
    "max_chars": 250,
    "formato": "wav",          # wav | mp3 | ambos
    "normalizar": True,
    "legenda_srt": True,
    "salvar_trechos": False,
    "marcacoes": True,         # interpreta {{emocao=0.8}} etc. no roteiro
    "transcrever": False,      # depois de narrar, transcreve com o Whisper (tempo de cada palavra) -> <nome>.transcricao.json
}

WHISPER_MODELO = os.getenv("WHISPER_MODELO", "turbo")
_whisper = None


def transcrever(caminho: Path, idioma: str, legendas: list) -> list:
    """Mesmo formato do whisper_transcribe.py do site: [{start, end, text, words: [{word, start, end}]}]."""
    global _whisper
    if MOCK:  # palavras espalhadas dentro de cada trecho narrado
        segs = []
        for a, b, t in legendas:
            ws = t.split()
            passo = (b - a) / max(1, len(ws))
            segs.append({"start": round(a, 3), "end": round(b, 3), "text": t,
                         "words": [{"word": w, "start": round(a + i * passo, 3), "end": round(a + (i + 1) * passo, 3)}
                                   for i, w in enumerate(ws)]})
        return segs
    if _whisper is None:
        import whisper

        store.heartbeat(TIPO, "carregando_modelo", modelo=f"whisper-{WHISPER_MODELO}")
        log(f"carregando whisper {WHISPER_MODELO} ...")
        _whisper = whisper.load_model(WHISPER_MODELO, device=DEVICE)
    res = _whisper.transcribe(str(caminho), language=(idioma or "pt").split("-")[0], word_timestamps=True,
                              verbose=False)
    segs = []
    for sg in res.get("segments", []):
        texto = str(sg.get("text") or "").strip()
        if not texto:
            continue
        segs.append({"start": float(sg["start"]), "end": float(sg["end"]), "text": texto,
                     "words": [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
                               for w in sg.get("words", []) if str(w.get("word", "")).strip()]})
    return segs

modelos: "OrderedDict[str, object]" = OrderedDict()
conds_padrao: dict = {}


def log(msg: str) -> None:
    print(f"[{store.agora()}] [voz] {msg}", flush=True)


# ------------------------------------------------------------ escolha de engine


def resolver_engine(cfg: dict) -> tuple[str, str]:
    """Retorna (chave_engine, language_id)."""
    modelo = str(cfg.get("modelo", "auto")).lower()
    idioma = str(cfg.get("idioma", "pt-br")).lower().replace("_", "-")
    base = idioma.split("-")[0]

    if modelo == "turbo" or (modelo == "auto" and base == "en"):
        return "turbo", "en"
    if modelo in ("auto", "pacote"):
        loc = APELIDOS_PACOTE.get(idioma)
        if loc:
            return f"pacote:{loc}", PACOTES[loc][1]
        if modelo == "pacote":
            raise ValueError(f"não existe pacote dedicado para '{idioma}'. Pacotes: {', '.join(PACOTES)}")
    if modelo not in ("auto", "pacote", "multilingual"):
        raise ValueError("modelo deve ser auto, pacote, turbo ou multilingual")
    if base not in IDIOMAS_MTL:
        raise ValueError(f"idioma '{idioma}' não suportado. Suportados: {', '.join(sorted(IDIOMAS_MTL))}")
    return "multilingual", base


# ------------------------------------------------------------ carregamento


def _montar_pasta_pacote(loc: str) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    repo, _ = PACOTES[loc]
    base = Path(snapshot_download(
        "ResembleAI/chatterbox",
        allow_patterns=["ve.pt", "conds.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"],
    ))
    pac = Path(snapshot_download(repo, allow_patterns=["t3_*.safetensors", "s3gen_v3.pt", "*.json"]))
    t3s = sorted(p.name for p in pac.glob("t3_*.safetensors"))
    if not t3s:
        raise RuntimeError(f"não achei o arquivo t3_*.safetensors em {repo}")

    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    pasta = hf_home / "studio_pacotes" / loc
    pasta.mkdir(parents=True, exist_ok=True)

    def ligar(nome_destino: str, origem: Path):
        destino = pasta / nome_destino
        if destino.is_symlink() or destino.exists():
            destino.unlink()
        destino.symlink_to(origem.resolve())

    for nome in ("ve.pt", "conds.pt", "Cangjie5_TC.json"):
        if (base / nome).exists():
            ligar(nome, base / nome)
    tok = "grapheme_mtl_merged_expanded_v1.json"
    ligar(tok, pac / tok if (pac / tok).exists() else base / tok)
    s3 = pac / "s3gen_v3.pt"
    ligar("s3gen.pt", s3 if (PACOTE_S3GEN == "v3" and s3.exists()) else base / "s3gen.pt")
    ligar(t3s[0], pac / t3s[0])
    return pasta, t3s[0]


def carregar(chave: str):
    if chave in modelos:
        modelos.move_to_end(chave)
        return modelos[chave]
    if MOCK:
        modelos[chave] = "mock"
        conds_padrao[chave] = "mock"
        return "mock"

    while len(modelos) >= MAX_MODELOS:
        antigo, _ = modelos.popitem(last=False)
        conds_padrao.pop(antigo, None)
        log(f"descarregando {antigo} para liberar VRAM")
        import torch

        torch.cuda.empty_cache()

    store.heartbeat(TIPO, "carregando_modelo", modelo=chave)
    log(f"carregando {chave} ...")
    t = time.time()
    if chave == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        m = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    elif chave == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        m = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model=MTL_VERSAO)
    elif chave.startswith("pacote:"):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        pasta, t3 = _montar_pasta_pacote(chave.split(":", 1)[1])
        m = ChatterboxMultilingualTTS.from_local(pasta, DEVICE, t3_model=t3)
    else:
        raise ValueError(f"engine desconhecida: {chave}")
    modelos[chave] = m
    conds_padrao[chave] = copy.copy(m.conds) if getattr(m, "conds", None) is not None else None
    log(f"{chave} pronto em {time.time() - t:.0f}s")
    return m


# ------------------------------------------------------------ áudio


def preparar_referencia(origem: Path, pasta_tmp: Path) -> Path:
    """Converte a voz de referência para wav mono 24 kHz (aceita mp3, m4a, ogg...)."""
    if not shutil.which("ffmpeg"):
        return origem
    pasta_tmp.mkdir(parents=True, exist_ok=True)
    destino = pasta_tmp / (origem.stem + "_ref.wav")
    if not destino.exists():
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(origem), "-ac", "1", "-ar", str(SR),
                            str(destino)], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"ffmpeg falhou na referência ({r.stderr.strip()[:200]}); usando arquivo original")
            return origem
    return destino


def aplicar_voz(chave: str, modelo, voz: Path | None, cfg: dict) -> None:
    if MOCK:
        return
    if voz is not None:
        ex = 0.0 if chave == "turbo" else float(cfg["exaggeration"])
        modelo.prepare_conditionals(str(voz), exaggeration=ex)
    else:
        if conds_padrao.get(chave) is None:
            raise ValueError("este modelo não tem voz padrão; coloque um áudio de referência no zip")
        modelo.conds = copy.copy(conds_padrao[chave])


def sintetizar(chave: str, modelo, texto: str, idioma: str, cfg: dict) -> np.ndarray:
    if MOCK:
        dur = max(0.4, len(texto) * 0.055)
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        time.sleep(0.05)
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if chave == "turbo":
        wav = modelo.generate(texto, temperature=float(cfg["temperature"]),
                              exaggeration=0.0, cfg_weight=0.0, min_p=0.0)
    else:
        wav = modelo.generate(texto, language_id=idioma, exaggeration=float(cfg["exaggeration"]),
                              cfg_weight=float(cfg["cfg_weight"]), temperature=float(cfg["temperature"]))
    return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)


def silencio(ms: float) -> np.ndarray:
    return np.zeros(int(SR * max(0.0, ms) / 1000), dtype=np.float32)


def tempo_srt(seg: float) -> str:
    ms = int(round(seg * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def salvar_audio(audio: np.ndarray, base: Path, formato: str) -> list[str]:
    import soundfile as sf

    wav = base.with_suffix(".wav")
    sf.write(wav, audio, SR, subtype="PCM_16")
    saidas = [wav.name]
    if formato in ("mp3", "ambos"):
        if shutil.which("ffmpeg"):
            mp3 = base.with_suffix(".mp3")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
            saidas.append(mp3.name)
            if formato == "mp3":
                wav.unlink()
                saidas.remove(wav.name)
        else:
            log("ffmpeg não encontrado; entregando só .wav")
    return saidas


# ------------------------------------------------------------ job


def ler_txt(p: Path) -> str:
    dados = p.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return dados.decode(enc)
        except UnicodeDecodeError:
            continue
    return dados.decode("utf-8", errors="replace")


def processar(st: dict) -> None:
    jid = st["id"]
    d = store.job_dir(jid)
    entrada, saida = d / "input", d / "output"
    params = store.get_params(jid)
    cfg = {**PADRAO, **(params.get("config") or {})}
    for k in ("exaggeration", "cfg_weight", "temperature", "pausa_ms", "pausa_paragrafo_ms", "max_chars"):
        cfg[k] = float(cfg[k])
    cfg["max_chars"] = int(min(max(cfg["max_chars"], 60), 600))

    chave, idioma = resolver_engine(cfg)
    avisos = []

    # mapa de vozes disponíveis no zip (nome do arquivo -> caminho)
    audios = {Path(a).name.lower(): entrada / a for a in params.get("audios", [])}
    audios.update({a.lower(): entrada / a for a in params.get("audios", [])})

    def achar_voz(nome) -> Path | None:
        if not nome:
            return None
        if str(nome).startswith("perfil:"):
            return perfis.clipes(str(nome)[7:])[0]  # o Chatterbox usa o 1º clipe do perfil (igual ao site)
        p = audios.get(str(nome).lower()) or audios.get(Path(str(nome)).name.lower())
        if p is None and perfis.existe(str(nome)):
            return perfis.clipes(str(nome))[0]
        if p is None:
            raise ValueError(f"voz '{nome}' não encontrada no zip")
        return preparar_referencia(p, d / "_ref")

    roteiros = [entrada / r for r in params["roteiros"]]
    planos = []
    for r in roteiros:
        eventos = montar_eventos(ler_txt(r), {**cfg, "voz": params.get("voz")}, avisos,
                                 marcacoes=bool(cfg.get("marcacoes", True)))
        planos.append((r, eventos))
    total = sum(1 for _, ev in planos for e in ev if e["tipo"] == "fala")
    if total == 0:
        raise ValueError("os roteiros estão vazios")

    store.update_status(jid, progresso={"feito": 0, "total": total}, avisos=avisos,
                        mensagem=f"carregando {chave}", engine=chave, idioma=idioma)
    modelo = carregar(chave)
    store.update_status(jid, mensagem="narrando")

    arquivos, erros, resumo = [], [], []
    feito = 0
    usar_nome = cfg.get("nome") and len(planos) == 1
    for r, eventos in planos:
        nome_base = store.slug(str(cfg["nome"]), 60) if usar_nome else store.slug(r.stem, 60)
        pedacos, legendas = [], []
        cursor = 0.0
        voz_atual = object()
        trechos_dir = saida / f"{nome_base}_trechos"
        for n, ev in enumerate(eventos):
            if store.cancel_requested(jid):
                store.update_status(jid, status=store.CANCELADO, concluido=store.agora(),
                                    mensagem="cancelado", arquivos=arquivos)
                return
            if ev["tipo"] == "silencio":
                pedacos.append(silencio(ev["ms"]))
                cursor += ev["ms"] / 1000
                continue
            c = ev["cfg"]
            if c.get("voz") != voz_atual:
                aplicar_voz(chave, modelo, achar_voz(c.get("voz")), c)
                voz_atual = c.get("voz")
            store.heartbeat(TIPO, "ocupado", job=jid, trecho=feito + 1, total=total)
            try:
                audio = sintetizar(chave, modelo, ev["texto"], idioma, c)
            except Exception as e:
                traceback.print_exc()
                erros.append({"roteiro": r.name, "trecho": ev["texto"][:80], "erro": str(e)[:300]})
                audio = np.zeros(0, dtype=np.float32)
            if audio.size:
                dur = audio.size / SR
                legendas.append((cursor, cursor + dur, ev["texto"]))
                pedacos.append(audio)
                cursor += dur
                if cfg.get("salvar_trechos"):
                    import soundfile as sf

                    trechos_dir.mkdir(exist_ok=True)
                    sf.write(trechos_dir / f"{len(legendas):04d}.wav", audio, SR, subtype="PCM_16")
            gap = c["pausa_paragrafo_ms"] if ev["fim_paragrafo"] else c["pausa_ms"]
            if n < len(eventos) - 1:
                pedacos.append(silencio(gap))
                cursor += gap / 1000
            feito += 1
            store.update_status(jid, progresso={"feito": feito, "total": total}, erros=erros)

        if not pedacos:
            continue
        final = np.concatenate(pedacos)
        if cfg.get("normalizar") and final.size:
            pico = float(np.max(np.abs(final)))
            if pico > 0:
                final = final * (0.89 / pico)  # pico em -1 dBFS
        arquivos += salvar_audio(final, saida / nome_base, cfg["formato"])
        if cfg.get("transcrever"):
            store.update_status(jid, mensagem="transcrevendo (Whisper)")
            store.heartbeat(TIPO, "ocupado", job=jid, etapa="transcrevendo")
            audio_feito = saida / f"{nome_base}.wav"
            if not audio_feito.exists():
                audio_feito = saida / f"{nome_base}.mp3"
            try:
                segs = transcrever(audio_feito, idioma, legendas)
                store.write_json(saida / f"{nome_base}.transcricao.json", segs)
                arquivos.append(f"{nome_base}.transcricao.json")
            except Exception as e:
                traceback.print_exc()
                erros.append({"roteiro": r.name, "trecho": "(transcrição)", "erro": f"Whisper: {str(e)[:300]}"})
        if cfg.get("legenda_srt") and legendas:
            srt = "\n".join(f"{i}\n{tempo_srt(a)} --> {tempo_srt(b)}\n{t}\n"
                            for i, (a, b, t) in enumerate(legendas, 1))
            (saida / f"{nome_base}.srt").write_text(srt, encoding="utf-8")
            arquivos.append(f"{nome_base}.srt")
        if cfg.get("salvar_trechos") and trechos_dir.exists():
            arquivos += [f"{trechos_dir.name}/{p.name}" for p in sorted(trechos_dir.glob("*.wav"))]
        resumo.append({"roteiro": r.name, "arquivo": nome_base, "duracao_s": round(final.size / SR, 1),
                       "trechos": len(legendas)})
        store.update_status(jid, arquivos=arquivos)

    store.write_json(saida / "resumo.json", {"engine": chave, "idioma": idioma, "config": cfg,
                                              "roteiros": resumo, "erros": erros, "avisos": avisos})
    arquivos.append("resumo.json")
    status = store.CONCLUIDO if not erros else (store.CONCLUIDO_COM_ERROS if resumo else store.ERRO)
    dur_total = sum(x["duracao_s"] for x in resumo)
    store.update_status(jid, status=status, concluido=store.agora(), arquivos=arquivos,
                        mensagem=f"{len(resumo)} áudio(s), {dur_total / 60:.1f} min no total")
    log(f"{jid}: {len(resumo)} áudio(s), {dur_total:.0f}s de fala ({chave}, {idioma})")


def main() -> None:
    store.ensure_dirs()
    store.requeue_stale(TIPO)
    log(f"iniciando (mock={MOCK}, dados={store.DATA_DIR}, pré-carregar={PRELOAD})")
    for chave in PRELOAD:
        try:
            carregar(chave)
        except Exception:
            traceback.print_exc()
            log(f"falha ao pré-carregar {chave}")
    while True:
        st = store.claim_next(TIPO)
        if not st:
            store.heartbeat(TIPO, "pronto", modelos_carregados=list(modelos))
            time.sleep(1)
            continue
        log(f"iniciando {st['id']}")
        try:
            processar(st)
        except Exception as e:
            traceback.print_exc()
            store.update_status(st["id"], status=store.ERRO, concluido=store.agora(), mensagem=str(e)[:500])


if __name__ == "__main__":
    main()
