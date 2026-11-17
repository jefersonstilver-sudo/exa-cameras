#!/usr/bin/env python3
"""EXA Mídia — confere se cada câmera do Portal da Cidade está apontada para o lugar de sempre (17/11/2026).

Compara um quadro atual da câmera com a imagem de referência dela (aponta/ref/<camera>.jpg) por pontos fixos
(ORB + RANSAC, transformação afim parcial — o mesmo método do alinhamento da fila). Câmera no lugar: muitos pontos
casam e a transformação é pequena (deslize de alguns pixels). Câmera virada para outro lugar: quase nada casa.

Uso: conferir.py <camera> <quadro.jpg> [quadro_anterior.jpg]   → imprime um JSON {"aponta": "ok"|"fora"|"noite"|"sem_ref"|"erro", ...}
Só julga de dia (07h00–18h30 de Brasília): à noite a imagem muda demais e daria falso "fora".
"""
import json, os, sys, time
from datetime import datetime, timedelta, timezone

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
# câmera → referência (as três da fila usam as mesmas imagens do alinhamento em fila/ref)
REFS = {
    "fozpontedaamizadesentidoparaguai": "../../fila/ref/ponte.jpg",
    "fozaduanapontedaamizade": "../../fila/ref/aduana.jpg",
    "fozsentidopontedaamizade01": "../../fila/ref/br277.jpg",
    "fozpontetancredoneves": "fozpontetancredoneves.jpg",
    "fozpontedaintegracao": "fozpontedaintegracao.jpg",
    "fozpontedaamizadesentidobrasil": "fozpontedaamizadesentidobrasil.jpg",
}
MIN_PONTOS = 25          # o alinhamento da fila aceita a partir de 25 pontos
MAX_ESCALA = 0.10        # zoom de até ±10%
MAX_DESLOC = 250         # deslize de até 250 px (em 1920x1080)

def sai(**k):
    k.setdefault("em", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(json.dumps(k, ensure_ascii=False)); sys.exit(0)

def main():
    if len(sys.argv) < 3: sai(aponta="erro", erro="uso: conferir.py camera quadro.jpg")
    cam, quadro = sys.argv[1], sys.argv[2]
    br = datetime.now(timezone.utc) - timedelta(hours=3)
    if not (7 * 60 <= br.hour * 60 + br.minute <= 18 * 60 + 30): sai(aponta="noite")
    ref = REFS.get(cam)
    caminho = os.path.normpath(os.path.join(PASTA, ref)) if ref else None
    if not caminho or not os.path.exists(caminho):
        # sem referência: ao menos avisa quando o enquadramento muda entre duas conferências (câmera girou)
        ant = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
        if not ant: sai(aponta="sem_ref")
        caminho = ant; SEM_REF = True
    else: SEM_REF = False
    try:
        import cv2, numpy as np
        a = cv2.imread(quadro, cv2.IMREAD_GRAYSCALE); r = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if a is None or r is None: sai(aponta="erro", erro="imagem ilegível")
        r = cv2.resize(r, (1920, 1080)); lim = MAX_DESLOC
        if a.shape[1] < 1000:
            # quadro tirado do clipe das TV boxes (clipes.yml: corte 1620x1080 no centro → 720x480): a referência passa pelo mesmo corte
            r = cv2.resize(r[:, 150:1770], (a.shape[1], a.shape[0])); lim = MAX_DESLOC * a.shape[1] / 1620
        else:
            a = cv2.resize(a, (1920, 1080))
        if float(a.std()) < 6: sai(aponta="fora", motivo="imagem lisa (preta, branca ou sem sinal)", pontos=0)
        orb = cv2.ORB_create(4000 if a.shape[1] >= 1000 else 2500)
        ka, da = orb.detectAndCompute(a, None); kr, dr = orb.detectAndCompute(r, None)
        if da is None or dr is None or len(ka) < 40: sai(aponta="fora", motivo="sem detalhes na imagem", pontos=0)
        m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, dr), key=lambda x: x.distance)[:800]
        if len(m) < 10: sai(aponta="fora", motivo="nada casa com a referência", pontos=0)
        M, inl = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kr[x.trainIdx].pt for x in m]),
                                             method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000)
        n = int(inl.sum()) if inl is not None else 0
        if M is None: sai(aponta="fora", motivo="nada casa com a referência", pontos=n)
        esc = float(np.hypot(M[0, 0], M[1, 0])); dx, dy = float(M[0, 2]), float(M[1, 2])
        ok = n >= MIN_PONTOS and abs(esc - 1) <= MAX_ESCALA and abs(dx) <= lim and abs(dy) <= lim
        if SEM_REF: sai(aponta="sem_ref" if ok else "mudou", pontos=n, **({} if ok else {"motivo": "o enquadramento mudou desde a conferência anterior"}))
        sai(aponta="ok" if ok else "fora", pontos=n, escala=round(esc, 3), dx=round(dx), dy=round(dy),
            **({} if ok else {"motivo": "pouca coisa casa com a referência" if n < MIN_PONTOS else "enquadramento mudou (zoom ou giro)"}))
    except SystemExit: raise
    except Exception as e:
        sai(aponta="erro", erro=str(e)[:120])

main()
