"""Worker de imagens: pega jobs 'imagem' da fila e gera com Z-Image Turbo."""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import store  # noqa: E402

TIPO = "imagem"
MOCK = os.getenv("MOCK") == "1"
MODEL_ID = os.getenv("IMG_MODEL", "Tongyi-MAI/Z-Image-Turbo")
OFFLOAD = os.getenv("IMG_CPU_OFFLOAD", "auto")  # auto | 1 | 0
PRELOAD = os.getenv("PRELOAD", "1") == "1"

pipe = None


def log(msg: str) -> None:
    print(f"[{store.agora()}] [imagem] {msg}", flush=True)


def carregar() -> None:
    global pipe
    if pipe is not None or MOCK:
        return
    store.heartbeat(TIPO, "carregando_modelo", modelo=MODEL_ID)
    log(f"carregando {MODEL_ID} ...")
    t = time.time()
    import torch
    from diffusers import ZImagePipeline

    p = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    usar_offload = OFFLOAD == "1" or (OFFLOAD == "auto" and vram < 30)
    if usar_offload:
        p.enable_model_cpu_offload()
    else:
        p.to("cuda")
    pipe = p
    log(f"modelo pronto em {time.time() - t:.0f}s (VRAM {vram:.0f} GB, offload={'sim' if usar_offload else 'não'})")


def gerar(item: dict):
    if MOCK:
        from PIL import Image, ImageDraw

        cor = ((item["seed"] * 37) % 255, (item["seed"] * 91) % 255, (item["seed"] * 13) % 255)
        img = Image.new("RGB", (item["largura"], item["altura"]), cor)
        ImageDraw.Draw(img).text((20, 20), item["prompt_original"][:80], fill=(255, 255, 255))
        time.sleep(0.2)
        return img
    import torch

    gen = torch.Generator("cuda").manual_seed(int(item["seed"]))
    return pipe(
        prompt=item["prompt"],
        height=item["altura"],
        width=item["largura"],
        num_inference_steps=item["passos"],
        guidance_scale=0.0,  # Turbo não usa CFG
        generator=gen,
    ).images[0]


def salvar(img, caminho: Path, formato: str) -> None:
    if formato == "jpg":
        img.convert("RGB").save(caminho, quality=95)
    elif formato == "webp":
        img.save(caminho, quality=95)
    else:
        img.save(caminho)


def processar(st: dict) -> None:
    jid = st["id"]
    d = store.job_dir(jid)
    params = store.get_params(jid)
    itens = params["itens"]
    formato = params.get("formato", "png")
    total = len(itens)
    store.update_status(jid, progresso={"feito": 0, "total": total}, mensagem="carregando modelo")
    carregar()
    store.update_status(jid, mensagem="gerando")

    arquivos, erros, meta = [], [], []
    t0 = time.time()
    for i, it in enumerate(itens):
        if store.cancel_requested(jid):
            store.update_status(jid, status=store.CANCELADO, concluido=store.agora(),
                                mensagem=f"cancelado após {i} de {total}")
            log(f"{jid} cancelado")
            return
        store.heartbeat(TIPO, "ocupado", job=jid, item=i + 1, total=total)
        try:
            t = time.time()
            img = gerar(it)
            nome = f"{it['nome']}.{formato}"
            salvar(img, d / "output" / nome, formato)
            arquivos.append(nome)
            meta.append({**it, "arquivo": nome, "segundos": round(time.time() - t, 2)})
        except Exception as e:  # segue para o próximo prompt
            traceback.print_exc()
            erros.append({"item": it["nome"], "erro": str(e)[:500]})
            if "out of memory" in str(e).lower() and not MOCK:
                import torch

                torch.cuda.empty_cache()
        store.update_status(jid, progresso={"feito": i + 1, "total": total}, arquivos=arquivos, erros=erros)

    store.write_json(d / "output" / "metadados.json", meta)
    arquivos.append("metadados.json")
    final = store.CONCLUIDO if not erros else (store.CONCLUIDO_COM_ERROS if meta else store.ERRO)
    dur = time.time() - t0
    store.update_status(jid, status=final, concluido=store.agora(), arquivos=arquivos,
                        mensagem=f"{len(meta)} imagens em {dur:.0f}s")
    log(f"{jid}: {len(meta)}/{total} imagens em {dur:.0f}s")


def main() -> None:
    store.ensure_dirs()
    store.requeue_stale(TIPO)
    log(f"iniciando (mock={MOCK}, dados={store.DATA_DIR})")
    if PRELOAD:
        try:
            carregar()
        except Exception:
            traceback.print_exc()
            log("falha ao pré-carregar; vou tentar de novo no primeiro job")
    while True:
        st = store.claim_next(TIPO)
        if not st:
            store.heartbeat(TIPO, "pronto" if (pipe is not None or MOCK) else "ocioso", modelo=MODEL_ID)
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
