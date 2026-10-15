"""Leitura do roteiro: divide em frases curtas e interpreta marcações {{ ... }}.

Marcações aceitas (valem do ponto onde aparecem em diante):
    {{emocao=0.8}}            mesmo que exaggeration (0.25 a 2.0; padrão 0.5)
    {{cfg=0.3}}               cfg_weight (0 a 1; menor = fala mais calma)
    {{temperatura=0.7}}
    {{pausa=300}}             silêncio entre frases (ms)
    {{pausa_paragrafo=900}}   silêncio entre parágrafos (ms)
    {{voz=outra.wav}}         troca a voz clonada (arquivo dentro do zip)
    {{perfil=joao}}           troca para um perfil de voz guardado no pod
    {{silencio=1500}}         insere um silêncio uma única vez (ms)
Pode juntar várias: {{emocao=0.9, cfg=0.3}}
"""
import re

MARCA = re.compile(r"\{\{(.*?)\}\}", re.S)

ALIASES = {
    "exaggeration": "exaggeration", "exagero": "exaggeration", "emocao": "exaggeration", "emoção": "exaggeration",
    "cfg": "cfg_weight", "cfg_weight": "cfg_weight",
    "temperatura": "temperature", "temperature": "temperature",
    "pausa": "pausa_ms", "pausa_ms": "pausa_ms",
    "pausa_paragrafo": "pausa_paragrafo_ms", "pausa_paragrafo_ms": "pausa_paragrafo_ms",
    "voz": "voz", "perfil": "perfil",
    "silencio": "silencio", "silêncio": "silencio",
}
NUMERICOS = {"exaggeration", "cfg_weight", "temperature", "pausa_ms", "pausa_paragrafo_ms", "silencio"}

FIM_FRASE = re.compile(r"(?<=[.!?…;:])\s+")


def dividir_frases(texto: str, max_chars: int) -> list[str]:
    texto = " ".join(texto.split())
    if not texto:
        return []
    pedacos, atual = [], ""
    for frase in FIM_FRASE.split(texto):
        while len(frase) > max_chars:  # frase longa demais: corta em vírgula ou espaço
            corte = frase.rfind(", ", 0, max_chars)
            if corte < max_chars * 0.4:
                corte = frase.rfind(" ", 0, max_chars)
            if corte <= 0:
                corte = max_chars
            if atual:
                pedacos.append(atual)
                atual = ""
            pedacos.append(frase[: corte + 1].strip())
            frase = frase[corte + 1:].strip()
        if not frase:
            continue
        if not atual:
            atual = frase
        elif len(atual) + 1 + len(frase) <= max_chars:
            atual += " " + frase
        else:
            pedacos.append(atual)
            atual = frase
    if atual:
        pedacos.append(atual)
    return pedacos


def _ler_marca(conteudo: str, avisos: list) -> list[tuple[str, object]]:
    pares = []
    for parte in re.split(r"[,;\n]", conteudo):
        parte = parte.strip()
        if not parte:
            continue
        m = re.match(r"^([\wçãõéêáíóú]+)\s*[=:]\s*(.+)$", parte, re.I)
        if not m:
            avisos.append(f"marcação ignorada: {{{{{parte}}}}}")
            continue
        chave = ALIASES.get(m.group(1).lower())
        valor = m.group(2).strip().strip("\"'")
        if not chave:
            avisos.append(f"opção desconhecida: {m.group(1)}")
            continue
        if chave in NUMERICOS:
            try:
                valor = float(valor.replace(",", "."))
            except ValueError:
                avisos.append(f"valor inválido em {m.group(1)}={valor}")
                continue
        if chave == "perfil":
            chave, valor = "voz", f"perfil:{valor}"
        pares.append((chave, valor))
    return pares


def montar_eventos(texto: str, cfg_base: dict, avisos: list, marcacoes: bool = True) -> list[dict]:
    """Transforma o roteiro em uma lista de eventos 'fala' e 'silencio'."""
    cfg = dict(cfg_base)
    eventos = []
    blocos = MARCA.split(texto) if marcacoes else [texto]
    for i, bloco in enumerate(blocos):
        if i % 2 == 1:
            for chave, valor in _ler_marca(bloco, avisos):
                if chave == "silencio":
                    eventos.append({"tipo": "silencio", "ms": int(valor)})
                else:
                    cfg[chave] = valor
            continue
        for paragrafo in re.split(r"\n\s*\n", bloco):
            frases = dividir_frases(paragrafo, int(cfg["max_chars"]))
            for k, frase in enumerate(frases):
                eventos.append({"tipo": "fala", "texto": frase, "cfg": dict(cfg),
                                "fim_paragrafo": k == len(frases) - 1})
    return eventos
