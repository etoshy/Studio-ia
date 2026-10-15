#!/usr/bin/env python3
"""Cliente do Studio IA para usar no seu PC.

Precisa só de:  pip install requests

Defina a URL uma vez (ou use --url em cada comando):
    Windows:  set STUDIO_URL=https://xxxxx-8000.proxy.runpod.net
    Linux/Mac: export STUDIO_URL=https://xxxxx-8000.proxy.runpod.net

Exemplos:
    python studio.py imagens prompts.txt --proporcao 16:9 --estilo-depois ", cinematic, 35mm"
    python studio.py voz ep01/ ep02/ ingles.zip
    python studio.py voz roteiros/*            (cada pasta ou zip vira um job)
    python studio.py vozes-importar perfis_exportados_do_site.zip
    python studio.py vozes
    python studio.py status
    python studio.py baixar img-20261015-143000-ab12cd
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Instale antes:  pip install requests")


def api(args):
    url = (args.url or os.getenv("STUDIO_URL") or "").rstrip("/")
    if not url:
        sys.exit("Informe a URL com --url ou na variável STUDIO_URL")
    s = requests.Session()
    chave = args.chave or os.getenv("STUDIO_KEY")
    if chave:
        s.headers["X-API-Key"] = chave
    return url, s


def checar(r):
    if r.status_code >= 400:
        try:
            detalhe = r.json()
        except Exception:
            detalhe = r.text[:500]
        sys.exit(f"Erro {r.status_code}: {detalhe}")
    return r


def esperar(url, s, ids, intervalo=3):
    pendentes = set(ids)
    finais = {"concluido", "concluido_com_erros", "erro", "cancelado"}
    ultimo = {}
    while pendentes:
        for jid in list(pendentes):
            try:
                st = s.get(f"{url}/jobs/{jid}", timeout=30).json()
            except Exception as e:
                print(f"  (falha ao consultar {jid}: {e}; tentando de novo)")
                continue
            p = st.get("progresso", {})
            linha = f"{jid}  {st['status']:<20} {p.get('feito', 0)}/{p.get('total', 0)}  {st.get('mensagem', '')}"
            if ultimo.get(jid) != linha:
                print(linha)
                ultimo[jid] = linha
            if st["status"] in finais:
                pendentes.discard(jid)
                for e in st.get("erros", [])[:5]:
                    print(f"    erro: {e}")
        if pendentes:
            time.sleep(intervalo)


def baixar(url, s, jid, saida: Path):
    r = checar(s.get(f"{url}/jobs/{jid}/zip", timeout=600))
    destino = saida / jid
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(destino)
    print(f"  salvo em {destino}")
    return destino


def cmd_imagens(args):
    url, s = api(args)
    texto = Path(args.arquivo).read_text(encoding="utf-8-sig")
    prompts = [l.strip() for l in texto.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        sys.exit("nenhum prompt no arquivo")
    corpo = {
        "prompts": prompts,
        "nome": args.nome or Path(args.arquivo).stem,
        "proporcao": args.proporcao,
        "passos": args.passos,
        "quantidade": args.quantidade,
        "formato": args.formato,
        "estilo_antes": args.estilo_antes,
        "estilo_depois": args.estilo_depois,
    }
    if args.seed is not None:
        corpo["seed"] = args.seed
    st = checar(s.post(f"{url}/imagens", json=corpo, timeout=60)).json()
    print(f"job criado: {st['id']}  ({st['progresso']['total']} imagens)")
    if args.sem_esperar:
        return
    esperar(url, s, [st["id"]])
    baixar(url, s, st["id"], Path(args.saida))


def zipar(caminho: Path) -> tuple[str, bytes]:
    if caminho.is_file() and caminho.suffix.lower() == ".zip":
        return caminho.name, caminho.read_bytes()
    if caminho.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(caminho.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(caminho).as_posix())
        return caminho.name + ".zip", buf.getvalue()
    raise ValueError(f"{caminho} não é pasta nem .zip")


def cmd_voz(args):
    url, s = api(args)
    arquivos = []
    for c in args.itens:
        try:
            arquivos.append(zipar(Path(c)))
        except ValueError as e:
            print(f"ignorado: {e}")
    if not arquivos:
        sys.exit("nada para enviar")
    files = [("arquivos", (nome, dados, "application/zip")) for nome, dados in arquivos]
    data = {"config": args.config} if args.config else {}
    resp = checar(s.post(f"{url}/voz", files=files, data=data, timeout=600)).json()
    for e in resp.get("erros", []):
        print(f"recusado: {e['arquivo']}: {e['erro']}")
    ids = [j["id"] for j in resp["jobs"]]
    for j in resp["jobs"]:
        print(f"job criado: {j['id']}  ({j.get('nome')})")
    if args.sem_esperar or not ids:
        return
    esperar(url, s, ids)
    for jid in ids:
        baixar(url, s, jid, Path(args.saida))


def cmd_status(args):
    url, s = api(args)
    print(json.dumps(checar(s.get(f"{url}/status", timeout=30)).json(), ensure_ascii=False, indent=2))
    for j in checar(s.get(f"{url}/jobs", params={"limite": 15}, timeout=30)).json():
        p = j.get("progresso") or {}
        print(f"{j['id']}  {j['tipo']:<6} {j['status']:<20} {p.get('feito', 0)}/{p.get('total', 0)}  {j.get('nome') or ''}")


def cmd_vozes(args):
    url, s = api(args)
    for p in checar(s.get(f"{url}/vozes", timeout=30)).json():
        info = p.get("info") or {}
        extra = " · ".join(x for x in (info.get("genero"), info.get("idade"), info.get("descricao")) if x)
        print(f"{p['nome']:<25} {len(p['clipes'])} clipe(s)  {extra}")


def cmd_vozes_importar(args):
    url, s = api(args)
    for z in args.zips:
        with open(z, "rb") as f:
            r = checar(s.post(f"{url}/vozes/importar", files={"arquivo": (Path(z).name, f, "application/zip")},
                              data={"nome": args.nome or "", "substituir": str(args.substituir).lower()}, timeout=600))
        for p in r.json()["importados"]:
            print(f"importado: {p['nome']} ({p['clipes']} clipe(s)){'  [renomeado]' if p.get('renomeado') else ''}")


def cmd_vozes_exportar(args):
    url, s = api(args)
    r = checar(s.get(f"{url}/vozes/exportar", params={"nomes": ",".join(args.nomes)} if args.nomes else None, timeout=600))
    Path(args.saida).write_bytes(r.content)
    print(f"salvo em {args.saida} (no site: 🗣️ Perfis de voz -> 📦 Importar perfis)")


def cmd_baixar(args):
    url, s = api(args)
    if args.esperar:
        esperar(url, s, args.ids)
    for jid in args.ids:
        baixar(url, s, jid, Path(args.saida))


def main():
    ap = argparse.ArgumentParser(description="Cliente do Studio IA (RunPod)")
    ap.add_argument("--url", help="URL da API (ou variável STUDIO_URL)")
    ap.add_argument("--chave", help="só se você ligou API_KEY no servidor (ou STUDIO_KEY)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("imagens", help="gera imagens a partir de um .txt (1 prompt por linha)")
    a.add_argument("arquivo")
    a.add_argument("--nome")
    a.add_argument("--proporcao", default="16:9", help="1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 21:9, 16:9-hd, 9:16-hd")
    a.add_argument("--passos", type=int, default=9)
    a.add_argument("--quantidade", type=int, default=1, help="variações por prompt")
    a.add_argument("--formato", default="png", choices=["png", "jpg", "webp"])
    a.add_argument("--seed", type=int)
    a.add_argument("--estilo-antes", default="")
    a.add_argument("--estilo-depois", default="")
    a.add_argument("--saida", default="resultados")
    a.add_argument("--sem-esperar", action="store_true")
    a.set_defaults(func=cmd_imagens)

    v = sub.add_parser("voz", help="envia pastas ou zips (voz + config.json + roteiro.txt)")
    v.add_argument("itens", nargs="+", help="pastas ou arquivos .zip; cada um vira um job")
    v.add_argument("--config", help='JSON aplicado a todos, ex: \'{"idioma":"en"}\'')
    v.add_argument("--saida", default="resultados")
    v.add_argument("--sem-esperar", action="store_true")
    v.set_defaults(func=cmd_voz)

    st = sub.add_parser("status", help="workers, GPU e últimos jobs")
    st.set_defaults(func=cmd_status)

    b = sub.add_parser("baixar", help="baixa o resultado de jobs pelo id")
    b.add_argument("ids", nargs="+")
    b.add_argument("--esperar", action="store_true", help="espera terminar antes de baixar")
    b.add_argument("--saida", default="resultados")
    b.set_defaults(func=cmd_baixar)

    vz = sub.add_parser("vozes", help="lista os perfis de voz guardados no pod")
    vz.set_defaults(func=cmd_vozes)
    vi = sub.add_parser("vozes-importar", help="manda perfis pro pod (o .zip do 📦 Exportar do site, ou zip de áudios)")
    vi.add_argument("zips", nargs="+")
    vi.add_argument("--nome", help="nome do perfil (zip só com áudios na raiz)")
    vi.add_argument("--substituir", action="store_true")
    vi.set_defaults(func=cmd_vozes_importar)
    ve = sub.add_parser("vozes-exportar", help="baixa os perfis do pod num .zip que o site importa")
    ve.add_argument("nomes", nargs="*")
    ve.add_argument("--saida", default="perfis_de_voz_pod.zip")
    ve.set_defaults(func=cmd_vozes_exportar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
