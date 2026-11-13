#!/usr/bin/env python3
"""EXA — fila da Ponte da Amizade (Brasil -> Paraguai) em tempo real. v0.4 (12/11/2026)

Lê as câmeras fixas do Portal da Cidade (ou gravações, para teste), detecta cada veículo (YOLOX-s, Apache 2.0,
via OpenCV), acompanha veículo a veículo e publica a cada PASSO segundos um resumo por trecho:
veículos, parados, estado (livre / lento / parado), velocidade e metros de fila (onde a câmera já tem escala) e confiança.
Regra do Jeferson: quando não dá para ver direito, NÃO publica número — o trecho sai como "indisponivel".

uso:  fila_worker.py --modelo yolox_s.onnx --saida atual.json [--arquivo cam=video.ts ...] [--segundos 300] [--passo 10]
"""
import base64, argparse, json, os, subprocess, sys, threading, time, urllib.request
import cv2, numpy as np

BASE = "https://video02.logicahost.com.br/portaldacidade/%s.stream/playlist.m3u8"
VEIC = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
PAR = 0.2   # segundos entre os dois quadros do par (5 quadros/s)

# Cada câmera fixa = um trecho. Coordenadas na imagem 1920x1080.
#  recortes: onde o detector olha (640x640)   eixo: sentido de quem vai para o Paraguai (na imagem)
#  zona: polígono das faixas de CARRO sentido Paraguai (caminhão em fila própria fica fora)
#  metros: tabela y -> metros até o fim do trecho (None = câmera ainda sem escala medida)
CAMERAS = {
    "ponte": dict(stream="fozpontedaamizadesentidoparaguai", nome="Ponte (ponta paraguaia)", capacidade=50, ref="ponte.jpg",
        # 13/11: o tabuleiro tem 3 faixas que mudam de sentido (às vezes 2 vão ao Paraguai, às vezes 1). O sentido de cada coluna é
        # aprendido pelo movimento (fluxo óptico) e guardado; só entram na conta as colunas que andam rumo ao Paraguai (sobem na imagem).
        deck=dict(ey=[250, 480, 760, 1080], el=[700, 760, 880, 930], ry=[250, 600, 1080], er=[790, 1010, 1420], ncol=10, py_padrao=[6, 7, 8, 9], faixas=3, cap_col=5),
        recortes=[(450, 100, 1090, 740), (700, 440, 1340, 1080), (1000, 440, 1640, 1080)],
        eixo=(-0.58, -0.81),
        zona=[(700, 250), (790, 250), (1010, 600), (1420, 1080), (930, 1080), (880, 760), (760, 480)],
        # escala provisória (±30%): perfil de tempo de travessia de 12/11 ancorado nos 132 m mirante -> cabeceira
        metros=[(202, 0), (255, 10), (330, 34), (450, 64), (600, 92), (742, 113), (900, 125), (1050, 134)]),
    "aduana": dict(stream="fozaduanapontedaamizade", nome="Curva da aduana", capacidade=45, ref="aduana.jpg",
        recortes=[(560, 0, 1200, 640), (200, 280, 840, 920), (700, 280, 1340, 920)],
        eixo=(-0.80, 0.60), ignora=(7,),
        zona=[(1075, 20), (1215, 20), (1150, 300), (900, 560), (420, 900), (60, 900), (640, 470), (1000, 250)],
        # escala provisória (±25%): comprimento dos carros na imagem (perto ~20 px/m, fundo ~7,6 px/m ao longo da pista)
        metros=[(20, 145), (64, 127), (117, 107), (200, 86), (313, 67), (387, 54), (493, 40), (640, 18), (800, 6), (900, 0)]),
    "br277": dict(stream="fozsentidopontedaamizade01", nome="BR-277 / viaduto", eixo=(0.0, -1.0), ref="br277.jpg",
        # 13/11: MODO DENSO. Nesta tele quase rasante a fila fica para-choque com para-choque e o detector acha 1 carro em cada 4.
        # Aqui não se conta caixa: mede-se, FAIXA POR FAIXA, quanto está ocupado (textura) e a velocidade (fluxo óptico em até 9 s).
        denso=True,
        # bordas das faixas sentido Paraguai: (y, x da borda esq., 1|2, 2|3, borda dir.)
        bordas=[(420, 940, 968, 1005, 1050), (450, 925, 962, 1005, 1075), (560, 900, 945, 1005, 1090), (700, 888, 945, 1015, 1118), (900, 888, 948, 1020, 1112), (1080, 888, 950, 1020, 1105)],
        # faixa da direita = fila quase só de caminhões no começo e carros no fim: passo médio maior
        faixas=[dict(nome="esquerda", passo_m=6.5), dict(nome="centro", passo_m=6.5), dict(nome="direita (caminhões)", passo_m=10.0, caminhoes=True)],
        # fora das 3 faixas, mas ainda fila sentido Paraguai: conta pelo detector = "pelo menos N" (ele deixa passar, não inventa)
        acessos=[dict(nome="curva à esquerda depois da junção (carros e caminhões)", lupa=True, zona=[(540, 255), (700, 250), (1060, 305), (1060, 400), (950, 452), (860, 447), (720, 407), (600, 377), (540, 362)]),
                 dict(nome="alça da direita", zona=[(1150, 610), (1280, 610), (1295, 855), (1340, 865), (1340, 910), (1100, 910), (1100, 862), (1158, 850)]),
                 dict(nome="acesso norte (junção)", zona=[(1072, 400), (1180, 400), (1190, 482), (1090, 492)])],
        recortes=[(520, 100, 1160, 740), (1000, 380, 1640, 1020)],
        zona=[(888, 1080), (888, 700), (900, 560), (925, 450), (940, 420), (1050, 420), (1075, 450), (1090, 560), (1118, 700), (1112, 900), (1105, 1080)],
        # escala: tracejado da pista = 2 m + 4 m (ciclo 6 m = 25 px em y=720 e 30 px em y=1040); confere com fila parada
        # (um carro a cada ~32 px = ~6,5 m). A leitura de 12/11 (ciclo de 12 m, 314 m) estava ERRADA: dobrava distância e velocidade.
        metros=[(420, 0), (500, 22), (600, 49), (700, 74), (800, 98), (900, 120), (1000, 141), (1080, 157)]),
}

class Detector:
    def __init__(s, caminho, lado=640):
        s.net = cv2.dnn.readNet(caminho); s.lado = lado
        g, st = [], []
        for k in (8, 16, 32):
            n = lado // k; yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            g.append(np.stack([xx, yy], -1).reshape(-1, 2)); st.append(np.full((n * n, 1), k))
        s.g = np.concatenate(g).astype(np.float32); s.st = np.concatenate(st).astype(np.float32)
    def _um(s, img):
        h, w = img.shape[:2]; r = min(s.lado / h, s.lado / w)
        tela = np.full((s.lado, s.lado, 3), 114, np.uint8); rs = cv2.resize(img, (int(w * r), int(h * r))); tela[:rs.shape[0], :rs.shape[1]] = rs
        s.net.setInput(cv2.dnn.blobFromImage(tela, 1.0, (s.lado, s.lado), swapRB=False)); o = s.net.forward()[0]
        xy = (o[:, :2] + s.g) * s.st; wh = np.exp(o[:, 2:4]) * s.st
        sc = o[:, 4:5] * o[:, 5:]; return np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r, sc.max(1), sc.argmax(1)
    def detectar(s, img, recortes, minimo=0.30):
        B, C, K = [], [], []
        for (x0, y0, x1, y1) in recortes:
            b, c, k = s._um(img[y0:y1, x0:x1]); m = (c > minimo) & np.isin(k, list(VEIC))
            b = b[m]; b[:, [0, 2]] += x0; b[:, [1, 3]] += y0; B.append(b); C.append(c[m]); K.append(k[m])
        B, C, K = np.concatenate(B), np.concatenate(C), np.concatenate(K)
        if not len(B): return B, C, K
        i = np.array(cv2.dnn.NMSBoxes([[float(a[0]), float(a[1]), float(a[2] - a[0]), float(a[3] - a[1])] for a in B], C.tolist(), minimo, 0.5)).reshape(-1)
        return B[i], C[i], K[i]

class Rastreador:
    """Acompanha cada veículo com previsão de movimento (velocidade constante). Identidade some após 4 s sem ver."""
    def __init__(s): s.t = {}; s.prox = 0
    def passo(s, agora, caixas, classes):
        cen = np.stack([(caixas[:, 0] + caixas[:, 2]) / 2, caixas[:, 3]], 1) if len(caixas) else np.zeros((0, 2))  # base da caixa = pé do veículo
        livres = set(range(len(cen)))
        for i, v in sorted(s.t.items(), key=lambda kv: -len(kv[1]["h"])):
            dt = agora - v["h"][-1][0]; prev = np.array(v["h"][-1][1:3]) + np.array(v["v"]) * dt
            raio = max(18.0, 0.7 * v["alt"]) + 0.5 * np.hypot(*v["v"]) * dt
            melhor, md = None, raio
            for j in livres:
                d = float(np.hypot(*(cen[j] - prev)))
                if d < md: melhor, md = j, d
            if melhor is not None:
                livres.discard(melhor); novo = cen[melhor]
                if dt > 0: v["v"] = list(0.6 * np.array(v["v"]) + 0.4 * (novo - np.array(v["h"][-1][1:3])) / dt)
                v["h"].append((agora, float(novo[0]), float(novo[1]))); v["h"] = v["h"][-40:]
                v["alt"] = float(caixas[melhor, 3] - caixas[melhor, 1]); v["k"] = int(classes[melhor]); v["visto"] = agora; v["cx"] = caixas[melhor].tolist()
        for j in livres:
            s.t[s.prox] = dict(h=[(agora, float(cen[j, 0]), float(cen[j, 1]))], v=[0.0, 0.0], alt=float(caixas[j, 3] - caixas[j, 1]), k=int(classes[j]), visto=agora, nasc=agora, cx=caixas[j].tolist()); s.prox += 1
        for i in [i for i, v in s.t.items() if agora - v["visto"] > 4.0]: del s.t[i]
        return s.t

def metros_de(tab, y):
    ys = [a for a, _ in tab]; ms = [b for _, b in tab]; return float(np.interp(y, ys, ms))

class Trecho:
    def __init__(s, chave, cfg):
        s.chave, s.cfg, s.r = chave, cfg, Rastreador(); s.zona = np.array(cfg["zona"], np.int32); s.eixo = np.array(cfg["eixo"]); s.amostras = []
        s.quadros = 0; s.ultimo = 0.0; s.brilho = []; s.nitidez = []; s.cont = []
    def alinhar(s, agora, img, img_ant):
        """A câmera do Portal desliza alguns pixels (13/11: até 47 px na vertical entre 11h41 e 15h). Todas as áreas e réguas foram
        desenhadas sobre uma imagem de referência; aqui cada quadro é levado de volta a ela (ORB + RANSAC, transformação afim parcial)."""
        if not s.cfg.get("ref"): return img, img_ant
        if not hasattr(s, "refimg"):
            s.refimg = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", s.cfg["ref"])); s.M = None; s.Mt = -1e9; s.reg = dict(ok=False)
            if s.refimg is not None:
                s.orb = cv2.ORB_create(4000); s.refk, s.refd = s.orb.detectAndCompute(cv2.cvtColor(s.refimg, cv2.COLOR_BGR2GRAY), None)
        if s.refimg is None: return img, img_ant
        if agora - s.Mt >= 20:
            s.Mt = agora
            try:
                kb, db = s.orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, s.refd), key=lambda x: x.distance)[:800]
                M, inl = cv2.estimateAffinePartial2D(np.float32([kb[x.queryIdx].pt for x in m]), np.float32([s.refk[x.trainIdx].pt for x in m]),
                                                     method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=4000)
                n = int(inl.sum()) if inl is not None else 0; esc = float(np.hypot(M[0, 0], M[1, 0])) if M is not None else 0
                if M is not None and n >= 25 and abs(esc - 1) < 0.08 and abs(M[0, 2]) < 200 and abs(M[1, 2]) < 200: s.M = M
                s.reg = dict(ok=s.M is not None, dx=None if s.M is None else round(float(s.M[0, 2]), 1), dy=None if s.M is None else round(float(s.M[1, 2]), 1), pontos=n)
            except Exception as e:
                s.reg = dict(ok=s.M is not None, erro=str(e)[:80])
        if s.M is None: return img, img_ant
        w = lambda q: None if q is None else cv2.warpAffine(q, s.M, (1920, 1080), borderMode=cv2.BORDER_REPLICATE)
        return w(img), w(img_ant)
    # ---------- sentido de cada coluna do tabuleiro da ponte ----------
    def _col(s, x, y):
        d = s.cfg["deck"]; xl = float(np.interp(y, d["ey"], d["el"])); xr = float(np.interp(y, d["ry"], d["er"]))
        u = (x - xl) / max(xr - xl, 1.0); return int(np.clip(u * d["ncol"], 0, d["ncol"] - 1)) if -0.05 <= u <= 1.05 else None
    def sentido(s, agora, g):
        d = s.cfg["deck"]
        if not hasattr(s, "colpy"):
            s.colpy = [c in d["py_padrao"] for c in range(d["ncol"])]; s.colt = [0.0] * d["ncol"]; s.gh = []; s.coldy = [None] * d["ncol"]
        s.gh.append((agora, g)); s.gh = [h for h in s.gh if h[0] >= agora - 4]
        velho = [h for h in s.gh if 1.5 <= agora - h[0] <= 4]
        if not velho: return
        t0, g0 = velho[0]; m = np.zeros_like(g); cv2.fillPoly(m, [s.zona], 255)
        def flx(msk, n):
            p = cv2.goodFeaturesToTrack(g0, n, 0.01, 5, mask=msk)
            if p is None: return np.zeros((0, 2)), np.zeros((0, 2))
            q, ok, _ = cv2.calcOpticalFlowPyrLK(g0, g, p, None, winSize=(21, 21), maxLevel=3)
            r, ok2, _ = cv2.calcOpticalFlowPyrLK(g, g0, q, None, winSize=(21, 21), maxLevel=3)
            bom = (ok.ravel() == 1) & (ok2.ravel() == 1) & (np.linalg.norm((r - p).reshape(-1, 2), axis=1) < 1.0)
            return p.reshape(-1, 2)[bom], (q - p).reshape(-1, 2)[bom] / (agora - t0)
        P, D = flx(m, 900)
        if not len(P): return
        # a câmera da ponte às vezes desliza devagar: o movimento do cenário parado (mato, prédios) é descontado de todos os pontos
        fundo = np.full_like(g, 255); cv2.fillPoly(fundo, [cv2.convexHull(np.array([(600, 240), (900, 240), (1500, 1080), (820, 1080)], np.int32))], 0)
        Pf, Df = flx(fundo, 400)
        s.desliza = [round(float(v), 2) for v in np.median(Df, 0)] if len(Df) >= 20 else [0.0, 0.0]
        D = D - np.array(s.desliza)
        for c in range(d["ncol"]):
            sel = [i for i in range(len(P)) if s._col(P[i, 0], P[i, 1]) == c and np.hypot(*D[i]) > 1.5]
            if len(sel) < 6: s.coldy[c] = None; continue
            dy = float(np.median(D[sel, 1])); s.coldy[c] = round(dy, 1)
            if abs(dy) >= 1.5: s.colpy[c] = dy < 0; s.colt[c] = agora      # sobe na imagem = afasta da câmera = rumo ao Paraguai
    def cap(s):
        d = s.cfg.get("deck")
        if d and hasattr(s, "colpy"): return d["cap_col"] * max(1, sum(s.colpy))
        return s.cfg.get("capacidade")
    def _foto(s, img):
        ok, b = cv2.imencode(".jpg", cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 50]); return base64.b64encode(b.tobytes()).decode() if ok else None
    def rapido(s, cinza_ant, cinza, caixa):
        """Velocidade instantânea (px/s no sentido Paraguai) de um veículo entre dois quadros a 0,2 s — não depende de identidade."""
        x0, y0, x1, y1 = [int(v) for v in caixa]; mx, my = int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)
        x0, y0, x1, y1 = max(x0 + mx, 0), max(y0 + my, 0), min(x1 - mx, 1919), min(y1 - my, 1079)
        if x1 - x0 < 6 or y1 - y0 < 6: return None
        pts = cv2.goodFeaturesToTrack(cinza[y0:y1, x0:x1], 12, 0.02, 3)
        if pts is None: return None
        pts = pts.reshape(-1, 1, 2) + np.array([x0, y0], np.float32)
        ant, ok, _ = cv2.calcOpticalFlowPyrLK(cinza, cinza_ant, pts.astype(np.float32), None, winSize=(21, 21), maxLevel=3)
        d = (pts - ant).reshape(-1, 2)[ok.reshape(-1) == 1]
        if len(d) < 3: return None
        d = np.median(d, 0); return float(d[0] * s.eixo[0] + d[1] * s.eixo[1]) / PAR, float(d[1])
    # ---------- modo denso (por faixa) ----------
    FAIXAS_Y = list(range(420, 1060, 40))
    def _denso_ini(s):
        B = np.array(s.cfg["bordas"], float); ys = np.arange(1080); nf = B.shape[1] - 2
        xs = [np.interp(ys, B[:, 0], B[:, i + 1]) for i in range(nf + 1)]
        s.dl = np.zeros((1080, 1920), np.uint8); xx = np.arange(1920)[None, :]
        for i in range(nf):
            m = (xx >= xs[i][:, None] + 3) & (xx < xs[i + 1][:, None] - 3) & (ys[:, None] >= 420); s.dl[m] = i + 1
        s.dx0, s.dx1 = int(B[:, 1].min()) - 8, int(B[:, -1].max()) + 8; s.dhist = []; s.dam = []; s.nf = nf
        s.dacc = [np.array(a["zona"], np.int32) for a in s.cfg.get("acessos", [])]; s.dac_cont = []
    def _fluxo(s, ga, gb, dt):
        """Por ponto: (y, faixa, velocidade px/s para cima = sentido Paraguai), com conferência ida e volta."""
        x0, x1 = s.dx0, s.dx1; a, b, m = ga[400:1080, x0:x1], gb[400:1080, x0:x1], s.dl[400:1080, x0:x1]
        p = cv2.goodFeaturesToTrack(a, 1200, 0.01, 4, mask=(m > 0).astype(np.uint8) * 255)
        z = np.zeros(0)
        if p is None: return z, z, z
        q, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
        r, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(21, 21), maxLevel=3)
        bom = (ok.ravel() == 1) & (ok2.ravel() == 1) & (np.linalg.norm((r - p).reshape(-1, 2), axis=1) < 1.0)
        P = p.reshape(-1, 2)[bom]; fx = m[P[:, 1].astype(int), P[:, 0].astype(int)]
        return P[:, 1] + 400, fx, -(q - p).reshape(-1, 2)[bom][:, 1] / dt
    def quadro_denso(s, agora, img, img_ant, det):
        if not hasattr(s, "dl"): s._denso_ini()
        s.quadros += 1; s.ultimo = agora
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); peq = cv2.resize(g, (480, 270)); s.brilho.append(float(peq.mean())); s.nitidez.append(float(cv2.Laplacian(peq, cv2.CV_64F).var()))
        x0, x1 = s.dx0, s.dx1; rec = g[:, x0:x1].astype(np.float32); forte = cv2.magnitude(cv2.Sobel(rec, cv2.CV_32F, 1, 0), cv2.Sobel(rec, cv2.CV_32F, 0, 1)) > 60; L = s.dl[:, x0:x1]
        # 13/11 15h: ocupação = pixel que NÃO parece asfalto. O asfalto de referência é medido no mesmo instante na pista contrária
        # (a luz muda: sol forte às 11h41, nublado às 15h — o critério "pixel escuro" zerava a contagem com céu nublado).
        gf = g.astype(np.float32); asf = gf[600:1000, 720:840]; amed = float(np.median(asf)); amad = float(np.median(np.abs(asf - amed))) + 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); smax = float(np.percentile(hsv[600:1000, 720:840, 1], 95))
        esc = (np.abs(gf - amed) > max(22.0, 4 * amad)) | (hsv[..., 1] > max(60.0, smax + 25)); E = esc[:, x0:x1]
        ocup = [[float(E[y:y + 40][L[y:y + 40] == f + 1].mean()) for y in s.FAIXAS_Y] for f in range(s.nf)]
        # grade: fatias de 40 px x blocos de 8 px em toda a largura da pista (inclui o zebrado, que vira coluna extra quando trava).
        # colunas de veículos na fatia = largura ocupada / largura de uma faixa; não depende de o motorista respeitar a faixa.
        Bd = np.array(s.cfg["bordas"], float); cc = []; ct = []; cel = []
        for y in s.FAIXAS_Y:
            ym = y + 20; xl = float(np.interp(ym, [420, 560, 700, 1080], [878, 878, 886, 886])); xs = float(np.interp(ym, Bd[:, 0], Bd[:, 3])); xr = float(np.interp(ym, Bd[:, 0], Bd[:, 4]))
            faixa_px = (xr - float(np.interp(ym, Bd[:, 0], Bd[:, 1]))) / 3.0; oc = ot = 0; lin = []
            x = int(xl)
            while x + 8 <= xr:
                if float(esc[y:y + 40, x:x + 8].mean()) >= 0.30:
                    lin.append(x)
                    if x + 4 < xs: oc += 8
                    else: ot += 8
                x += 8
            cc.append(min(oc / faixa_px, 3.0)); ct.append(min(ot / faixa_px, 1.2)); cel.append(lin)
        z = np.zeros(0)
        yr, fr_, vr = s._fluxo(cv2.cvtColor(img_ant, cv2.COLOR_BGR2GRAY), g, PAR) if img_ant is not None else (z, z, z)
        velho = [h for h in s.dhist if 2.0 <= agora - h[0] <= 9.5]
        yl, fl, vl = s._fluxo(velho[0][1], g, agora - velho[0][0]) if velho else (z, z, z)
        v = []
        for f in range(s.nf):
            vf = []
            for y in s.FAIXAS_Y:
                mpp = (metros_de(s.cfg["metros"], y + 40) - metros_de(s.cfg["metros"], y)) / 40.0
                a = vr[(yr >= y) & (yr < y + 40) & (fr_ == f + 1)]; b = vl[(yl >= y) & (yl < y + 40) & (fl == f + 1)]
                ra = float(np.median(a)) * mpp if len(a) >= 4 else None; lb = float(np.median(b)) * mpp if len(b) >= 4 else None
                vf.append(ra if (ra is not None and (abs(ra) >= 2.0 or lb is None)) else lb)
            v.append(vf)
        s.dam.append(dict(t=agora, ocup=ocup, v=v, n=int(len(yl) + len(yr)), cc=cc, ct=ct)); s.dam = [a for a in s.dam if a["t"] >= agora - 120]
        s.dhist.append((agora, g)); s.dhist = [h for h in s.dhist if h[0] >= agora - 10]
        if not hasattr(s, "ult_acc"): s.ult_acc = []
        if s.dacc and det is not None and s.quadros % 3 == 1:        # acessos: detector a cada 3 quadros
            b, c, k = det.detectar(img, s.cfg["recortes"], 0.25); n = [0] * len(s.dacc); cam = [0] * len(s.dacc)
            # lupa: a curva à esquerda depois da junção fica longe (carros de ~30 px): detector de novo na região ampliada 2x
            lp = [i for i, a in enumerate(s.cfg.get("acessos", [])) if a.get("lupa")]
            if lp:
                Z = np.concatenate([s.dacc[i] for i in lp]); X0, Y0 = max(int(Z[:, 0].min()) - 20, 0), max(int(Z[:, 1].min()) - 20, 0); X1, Y1 = min(int(Z[:, 0].max()) + 20, 1919), min(int(Z[:, 1].max()) + 20, 1079)
                G = cv2.resize(img[Y0:Y1, X0:X1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); H_, W_ = G.shape[:2]
                rec = [(x, y, min(x + 640, W_), min(y + 640, H_)) for y in range(0, max(H_ - 320, 1), 480) for x in range(0, max(W_ - 320, 1), 480)]
                b2, c2, k2 = det.detectar(G, rec, 0.22)
                if len(b2):
                    b2 = b2 / 2.0 + np.array([X0, Y0, X0, Y0]); fora = []
                    for j, bb in enumerate(b): 
                        pt = (float((bb[0] + bb[2]) / 2), float((bb[1] + bb[3]) / 2))
                        fora.append(not any(cv2.pointPolygonTest(s.dacc[i], pt, False) >= 0 for i in lp))
                    fora = np.array(fora, bool) if len(b) else np.zeros(0, bool)
                    b = np.concatenate([b[fora], b2]) if len(b) else b2; k = np.concatenate([k[fora], k2]) if len(k) else k2
            for bb, kk in zip(b, k):
                if int(kk) == 3: continue
                pt = (float((bb[0] + bb[2]) / 2), float((bb[1] + bb[3]) / 2))
                for i, zz in enumerate(s.dacc):
                    if cv2.pointPolygonTest(zz, pt, False) >= 0: n[i] += 1; cam[i] += int(int(kk) in (5, 7)); break
            s.dac_cont.append((agora, n, cam)); s.dac_cont = s.dac_cont[-40:]
            s.ult_acc = [[int(c) for c in bb] + [int(any(cv2.pointPolygonTest(zz, (float((bb[0] + bb[2]) / 2), float((bb[1] + bb[3]) / 2)), False) >= 0 for zz in s.dacc) and int(kk) != 3)] for bb, kk in zip(b, k)]
            s.ult_acc = [c for c in s.ult_acc if c[4]]
        s.ult = dict(t=agora, alinhamento=getattr(s, "reg", None), img=s._foto(img), caixas=s.ult_acc, fatias=[[int(o >= 0.30) for o in of] for of in ocup], fatias_y=s.FAIXAS_Y, celulas=cel, colunas_carros=[round(c, 1) for c in cc], colunas_direita=[round(c, 1) for c in ct])
    def resumo_denso(s, agora, janela, out):
        A = [a for a in s.dam if a["t"] >= agora - janela]
        if len(A) < 3: out["motivo"] = "poucos quadros"; return out
        tab = s.cfg["metros"]; total = max(m for _, m in tab); faixas = []
        for f, cf in enumerate(s.cfg["faixas"]):
            ocup = np.median(np.array([a["ocup"][f] for a in A]), 0)
            V = np.array([[np.nan if x is None else x for x in a["v"][f]] for a in A], float)
            with np.errstate(all="ignore"): vb = np.nanmedian(V, 0)
            lenta = (ocup >= 0.30) & ~(np.nan_to_num(vb, nan=0.0) * 3.6 >= 20)
            fim = 0; buraco = 0                 # fila = fatias ocupadas e lentas a partir do fim do trecho (topo), tolerando 1 buraco
            for i, ok in enumerate(lenta):
                if ok: fim = i + 1; buraco = 0
                else:
                    buraco += 1
                    if buraco > 1: break
            fila_m = (metros_de(tab, s.FAIXAS_Y[fim - 1] + 40) - metros_de(tab, s.FAIXAS_Y[0])) if fim else 0.0
            cheio_m = sum(metros_de(tab, s.FAIXAS_Y[i] + 40) - metros_de(tab, s.FAIXAS_Y[i]) for i in range(len(lenta)) if lenta[i])
            vv = vb[:fim][~np.isnan(vb[:fim])] if fim else vb[~np.isnan(vb)]
            vm = float(np.clip(np.median(vv), 0, None)) if len(vv) else None
            with np.errstate(all="ignore"): par = float(np.nanmean(np.abs(V[:, :fim]) < 0.15)) if fim else 0.0
            faixas.append(dict(nome=cf["nome"], ocupacao_pct=round(100 * float(ocup.mean())), fila_m=round(fila_m), kmh=None if vm is None else round(vm * 3.6, 1),
                               veiculos=round(cheio_m / cf["passo_m"]), cauda_alem=bool(fim >= len(s.FAIXAS_Y) - 1), parado_pct=round(100 * (0.0 if np.isnan(par) else par)), _v=vm, _fim=fim))
        acessos = []
        C = [c for c in s.dac_cont if c[0] >= agora - max(janela, 30)]
        for i, a in enumerate(s.cfg.get("acessos", [])):
            n = int(round(float(np.median([c[1][i] for c in C])))) if C else None; cm = int(round(float(np.median([c[2][i] for c in C])))) if C else None
            acessos.append(dict(nome=a["nome"], pelo_menos=n, caminhoes_ou_onibus=cm))
        fx_carros = [f for f, cf in zip(faixas, s.cfg["faixas"]) if not cf.get("caminhoes")]; pior = max(fx_carros, key=lambda f: f["fila_m"])
        vs = [f["_v"] for f in fx_carros if f["_v"] is not None and f["_fim"] >= 3] or [f["_v"] for f in fx_carros if f["_v"] is not None]
        v = float(np.median(vs)) if vs else None; fila_m = pior["fila_m"]; fim = pior["_fim"]
        pontos = float(np.mean([a["n"] for a in A])); nos_acessos = sum(a["pelo_menos"] or 0 for a in acessos)
        # contagem pela grade: soma, fatia a fatia, de colunas de veículos x metros da fatia ÷ (metros de pista por carro).
        # Calibração a olho em 3 quadros de 13/11 (11h41 sol, 14h04 sol, 15h00 nublado): 4,6 a 6,5 "coluna-metro" por carro -> 4,8 (±25%).
        # Faixa da direita (caminhões + carros): 10 m por veículo x 0,85.
        CC = np.median(np.array([a["cc"] for a in A]), 0); CT = np.median(np.array([a["ct"] for a in A]), 0)
        mf = [metros_de(tab, y + 40) - metros_de(tab, y) for y in s.FAIXAS_Y]
        carros = sum(c * m / 4.8 for c, m in zip(CC, mf) if c >= 0.6); direita = 0.85 * sum(c * m / 10.0 for c, m in zip(CT, mf) if c >= 0.5)
        nas_faixas = round(carros + direita)
        for f in faixas: f.pop("_v"); f.pop("_fim")
        out.update(carros_2_faixas=round(carros), veiculos_faixa_direita=round(direita), colunas_por_fatia=[round(float(c), 1) for c in CC], margem_pct=25, veiculos=nas_faixas + nos_acessos, veiculos_estimado=True, veiculos_nas_faixas=nas_faixas, veiculos_nos_acessos_min=nos_acessos, faixas=faixas, acessos=acessos,
                   ocupacao_pct=round(float(np.mean([f["ocupacao_pct"] for f in faixas]))), fila_m=fila_m if fim else None, kmh=None if v is None else round(v * 3.6, 1),
                   confianca=int(min(85, 45 + pontos / 10)), cauda_alem=any(f["cauda_alem"] for f in fx_carros), parado_pct=max(f["parado_pct"] for f in fx_carros))
        if fim < 3 or v is None: out["estado"] = "livre"
        else: out["estado"] = "parado" if v * 3.6 < 2.0 else ("lento" if v * 3.6 < 20 else "livre")
        if out["estado"] == "livre": out["fila_m"] = None; out["parado_pct"] = None
        elif v and v > 0.05: out["tempo_trecho_s"] = round(fila_m / v + (total - fila_m) / 11.0)
        return out
    def quadro(s, agora, img, det, img_ant=None):
        img, img_ant = s.alinhar(agora, img, img_ant)
        if s.cfg.get("denso"): return s.quadro_denso(agora, img, img_ant, det)
        b, c, k = det.detectar(img, s.cfg["recortes"]); vivos = s.r.passo(agora, b, k); s.quadros += 1; s.ultimo = agora
        cinza = cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY); s.brilho.append(float(cinza.mean())); s.nitidez.append(float(cv2.Laplacian(cinza, cv2.CV_64F).var()))
        na_zona = 0
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g0 = cv2.cvtColor(img_ant, cv2.COLOR_BGR2GRAY) if img_ant is not None else None
        if s.cfg.get("deck"): s.sentido(agora, g1)
        for i, v in vivos.items():
            if v["visto"] != agora or v["k"] == 3 or v["k"] in s.cfg.get("ignora", ()): continue                      # moto não entra na fila de carro
            x, y = v["h"][-1][1], v["h"][-1][2]
            if cv2.pointPolygonTest(s.zona, (x, y), False) < 0: continue
            if s.cfg.get("deck"):
                cc = s._col(x, y)
                if cc is None or not s.colpy[cc]: continue                                     # coluna de quem vem para o Brasil
            na_zona += 1
            alt = max(v["alt"], 8.0); rel = vm = None
            # 1) rápido: par de quadros a 0,2 s (vale para trânsito fluindo, quando não dá para acompanhar a identidade)
            rp = s.rapido(g0, g1, v["cx"]) if g0 is not None and "cx" in v else None
            if rp is not None and rp[0] / alt > 0.5:
                rel = rp[0] / alt
                if s.cfg["metros"]: vm = (metros_de(s.cfg["metros"], y - rp[1]) - metros_de(s.cfg["metros"], y)) / PAR
            else:
                # 2) lento: deslocamento do mesmo veículo em até 3 s (fila que anda devagar)
                ant = [h for h in v["h"] if h[0] >= agora - 3.0][0]; dt = agora - ant[0]
                if dt >= 0.9:
                    rel = ((x - ant[1]) * s.eixo[0] + (y - ant[2]) * s.eixo[1]) / dt / alt      # "tamanhos de veículo por segundo"
                    if s.cfg["metros"]: vm = (metros_de(s.cfg["metros"], ant[2]) - metros_de(s.cfg["metros"], y)) / dt
                elif rp is not None: rel = rp[0] / alt
            if rel is None: continue
            s.amostras.append(dict(t=agora, id=i, y=y, rel=rel, vm=vm, k=v["k"]))
        s.cont.append((agora, na_zona)); s.cont = s.cont[-200:]
        # para a página de monitor: o quadro analisado (reduzido) e cada veículo contado (1) ou não (0: fora da área, moto, ignorado)
        cx = []
        for i, v in vivos.items():
            if v["visto"] != agora or "cx" not in v: continue
            x, y = v["h"][-1][1], v["h"][-1][2]
            conta = int(v["k"] != 3 and v["k"] not in s.cfg.get("ignora", ()) and cv2.pointPolygonTest(s.zona, (x, y), False) >= 0)
            if conta and s.cfg.get("deck"): cc = s._col(x, y); conta = int(cc is not None and s.colpy[cc])
            cx.append([int(c) for c in v["cx"]] + [conta])
        s.ult = dict(t=agora, img=s._foto(img), caixas=cx, contados=sum(c[4] for c in cx), alinhamento=getattr(s, "reg", None))
        if s.cfg.get("deck"): s.ult.update(colunas_py=[int(v) for v in s.colpy], colunas_dy=s.coldy, deck=s.cfg["deck"], desliza=getattr(s, "desliza", None))
    def resumo(s, agora, janela):
        A = [a for a in s.amostras if a["t"] >= agora - janela]; s.amostras = [a for a in s.amostras if a["t"] >= agora - 120]
        out = dict(trecho=s.chave, nome=s.cfg["nome"], capacidade=s.cap(), trecho_m=(max(m for _, m in s.cfg["metros"]) if s.cfg["metros"] else None), estado="indisponivel", motivo=None, veiculos=None, parados=None, parado_pct=None, kmh=None, fila_m=None, tempo_trecho_s=None, confianca=0)
        if agora - s.ultimo > 15 or s.quadros < 3: out["motivo"] = "camera sem imagem"; return out
        br, nt = np.mean(s.brilho[-8:]), np.mean(s.nitidez[-8:])
        if br < 35: out["motivo"] = "escuro demais"; return out
        if nt < 25: out["motivo"] = "imagem sem nitidez (chuva, neblina ou camera fora de foco)"; return out
        if s.cfg.get("denso"): return s.resumo_denso(agora, janela, out)
        C = [n for t, n in s.cont if t >= agora - janela]
        if len(C) < 3: out["motivo"] = "poucos quadros"; return out
        na_zona = float(np.mean(C)); acompanhados = len(A) / len(C)
        rapidos = float((np.array([a["rel"] for a in A]) > 0.5).mean()) if A else 0.0     # fração andando de verdade
        por_quadro = na_zona
        rel = np.array([a["rel"] for a in A]) if A else np.zeros(0)
        parados = float((np.abs(rel) < 0.05).mean()) if len(rel) else 0.0
        out.update(veiculos=round(por_quadro, 1), parados=round(acompanhados * parados, 1), parado_pct=round(100 * acompanhados * parados / max(na_zona, 0.1)), rapidos_pct=round(rapidos * 100))
        if por_quadro < 2.0 or rapidos > 0.5: out.update(estado="livre", confianca=int(60 + 30 * min(1, rapidos / 0.8))); 
        else:
            med = float(np.median(rel))
            out["estado"] = "parado" if (parados > 0.6 or med < 0.03) else ("lento" if med < 0.45 else "livre")
            out["confianca"] = int(min(90, 40 + 5 * acompanhados))
        # fila exige densidade: poucos veículos passando devagar (lombada, curva) não são fila
        if s.cfg.get("deck") and hasattr(s, "colpy"):
            d = s.cfg["deck"]; out["faixas_rumo_py"] = int(np.clip(round(d["faixas"] * sum(s.colpy) / d["ncol"]), 1, d["faixas"] - 1))
        if s.cap() and out["estado"] != "livre" and por_quadro < 0.25 * s.cap(): out["estado"] = "livre"; out["motivo"] = "poucos veiculos na imagem"
        if s.cfg["metros"] and len(A) >= 8:
            vm = np.array([a["vm"] for a in A if a["vm"] is not None]); v = float(np.clip(vm.mean(), 0, None))
            out["kmh"] = round(v * 3.6, 1)
            if out["estado"] != "livre" and v * 3.6 >= 20: out["estado"] = "livre"     # a 20 km/h ou mais não é fila, seja qual for o tamanho da caixa
            ys = sorted(a["y"] for a in A if a["t"] == max(x["t"] for x in A))
            if out["estado"] != "livre" and len(ys) >= 3:
                tab = s.cfg["metros"]; out["fila_m"] = round(abs(metros_de(tab, ys[-1]) - metros_de(tab, ys[0])))
                if v > 0.05: out["tempo_trecho_s"] = round(max(m for _, m in tab) / v)
        return out

# ---- Travessia inteira (viaduto -> ponta paraguaia): junta os 3 trechos vistos e estima os 2 trechos cegos pelos vizinhos.
# Comprimentos medidos no mapa/satélite (m). VAO_A = curva sem câmera; VAO_B = aduana + parte da ponte sem câmera fixa.
VAO_A, VAO_B, V_LIVRE = 685, 730, 11.0     # V_LIVRE = 40 km/h quando o trecho está livre
_MEM = dict(votos={}, estado={}, hist=[])
def _estavel(chave, bruto, n=18, liga=0.6, desliga=0.3):
    """Sim/não com memória de 3 min: só liga com 60% dos últimos votos e só desliga abaixo de 30% (não fica piscando)."""
    v = _MEM["votos"].setdefault(chave, []); v.append(bool(bruto)); del v[:-n]
    frac = sum(v) / len(v); atual = _MEM["estado"].get(chave, bool(bruto))
    atual = (frac >= desliga) if atual else (frac >= liga); _MEM["estado"][chave] = atual; return atual
def travessia(tr, estavel=True):
    t = {x["trecho"]: x for x in tr}
    out = dict(estado="indisponivel", motivo=None, fila_min_m=None, fila_max_m=None, tempo_min_s=None, tempo_max_s=None, cauda=None)
    if any(k not in t or t[k]["estado"] == "indisponivel" for k in ("ponte", "aduana", "br277")):
        out["motivo"] = "trecho sem leitura: " + ", ".join(k for k in ("ponte", "aduana", "br277") if k not in t or t[k]["estado"] == "indisponivel"); return out
    def _cheio_bruto(k):
        x = t[k]
        if x["estado"] not in ("lento", "parado"): return False
        if "cauda_alem" in x and x["cauda_alem"] is not None: return bool(x["cauda_alem"])
        # câmera de detector: cheia se a fila ocupa 60% do trecho OU se está parada com a imagem lotada de veículos
        return (x["fila_m"] or 0) >= 0.6 * x["trecho_m"] or (x["estado"] == "parado" and x.get("capacidade") and (x["veiculos"] or 0) >= 0.6 * x["capacidade"])
    F = {k: (_estavel("fila_" + k, t[k]["estado"] in ("lento", "parado")) if estavel else t[k]["estado"] in ("lento", "parado")) for k in ("ponte", "aduana", "br277")}
    CH = {k: F[k] and (_estavel("cheio_" + k, _cheio_bruto(k)) if estavel else _cheio_bruto(k)) for k in ("ponte", "aduana", "br277")}
    fila = lambda k: F[k]
    cheio = lambda k: CH[k]
    vel = lambda k: max((t[k]["kmh"] or 0) / 3.6, 0.3) if fila(k) else max((t[k]["kmh"] or 0) / 3.6, V_LIVRE)
    def t_trecho(k):      # só o pedaço em fila anda devagar; o resto do trecho anda livre
        if not fila(k): return t[k]["trecho_m"] / vel(k)
        f = min(t[k]["fila_m"] or t[k]["trecho_m"], t[k]["trecho_m"]); return f / vel(k) + (t[k]["trecho_m"] - f) / V_LIVRE
    tmin = tmax = sum(t_trecho(k) for k in ("ponte", "aduana", "br277"))
    fmin = fmax = 0; cauda = "sem fila na ponta paraguaia"; aberta = False
    P = {}                                     # quanto de fila em cada trecho cego: (mín, máx) em metros
    def vao(comp, baixo, cima, funil=False):
        # devolve (tempo_min, tempo_max, fila_min, fila_max, continua_para_cima)
        if not cheio(baixo): return comp / V_LIVRE, comp / V_LIVRE, 0, 0, False
        if fila(cima) and not funil:                      # fila nas duas pontas -> o meio também está em fila
            a, b = sorted((vel(baixo), vel(cima))); return comp / b, comp / a, comp, comp, True
        # cauda dentro do trecho cego. Vale também quando a ponta de cima é um FUNIL (junção do viaduto): a fila de lá nasce
        # da própria junção, não prova que a fila da ponte chegou até ela (13/11 11h41: viaduto travado e curva livre).
        return comp / V_LIVRE, comp / vel(baixo), 0, comp, False
    if fila("ponte"):
        fmin = fmax = t["ponte"]["fila_m"] or 0; cauda = "na ponte (visivel)"
        a, b, c, d, segue = vao(VAO_B, "ponte", "aduana"); tmin += a; tmax += b; fmin += c; fmax += d; P["B"] = (c, d)
        if cheio("ponte"): cauda = "entre a aduana e a ponte (sem camera)"
        if segue:
            fmin += t["aduana"]["fila_m"] or 0; fmax += t["aduana"]["fila_m"] or 0; cauda = "na curva da aduana (visivel)"
            a, b, c, d, segue = vao(VAO_A, "aduana", "br277", funil=True); tmin += a; tmax += b; fmin += c; fmax += d; P["A"] = (c, d)
            if cheio("aduana"): cauda = "na curva antes da aduana (sem camera)"
            if segue:
                fmin += t["br277"]["fila_m"] or 0; fmax += t["br277"]["fila_m"] or 0; cauda = "na BR-277 (visivel)"
                if cheio("br277"): aberta = True; cauda = "antes do viaduto (alem das cameras)"
        else: tmin += VAO_A / V_LIVRE; tmax += VAO_A / V_LIVRE
    else:
        tmin += (VAO_A + VAO_B) / V_LIVRE; tmax += (VAO_A + VAO_B) / V_LIVRE
    # Filas separadas (ex.: trava no viaduto, anda livre na curva e trava de novo na ponte): somam no tamanho total.
    # Se a fila do viaduto passa da câmera, o total vira "mais de" (sem máximo), porque ninguém vê onde ela termina.
    separadas = []
    if fila("br277") and not cauda.startswith(("na BR-277", "antes do viaduto")):
        f = t["br277"]["fila_m"] or 0; fmin += f; fmax += f; separadas.append(dict(onde="viaduto da BR-277", fila_m=round(f), alem_da_camera=bool(cheio("br277"))))
        if cheio("br277"): aberta = True
    if fila("aduana") and cauda.startswith(("sem fila", "na ponte", "entre a aduana")) :
        f = t["aduana"]["fila_m"] or 0; fmin += f; fmax += f; separadas.append(dict(onde="chegada a aduana", fila_m=round(f), alem_da_camera=False))
    out["filas_separadas"] = separadas
    # Onde está cada pedaço da fila, do viaduto até o Paraguai: o que é CONTADO por câmera e o que é CONTA de trecho sem câmera
    # (2 faixas, um carro a cada 6,5 m). Em trecho sem câmera a fila pode ser de 0 até o trecho inteiro.
    cam = lambda k, nome: dict(onde=nome, tipo="camera", fila_min_m=round(t[k]["fila_m"] or 0) if fila(k) else 0, fila_max_m=round(t[k]["fila_m"] or 0) if fila(k) else 0,
                               carros_min=round(t[k]["veiculos"] or 0), carros_max=round(t[k]["veiculos"] or 0))
    sem = lambda q, nome, comp: dict(onde=nome, tipo="sem camera", trecho_m=comp, fila_min_m=round(P.get(q, (0, 0))[0]), fila_max_m=round(P.get(q, (0, 0))[1]),
                                     carros_min=round(P.get(q, (0, 0))[0] * 2 / 6.5), carros_max=round(P.get(q, (0, 0))[1] * 2 / 6.5))
    out["partes"] = [cam("br277", "viaduto da BR-277"), sem("A", "curva depois do viaduto", VAO_A), cam("aduana", "chegada a aduana"),
                     sem("B", "aduana e inicio da ponte", VAO_B), cam("ponte", "ponta paraguaia da ponte")]
    out["carros_contados"] = sum(p["carros_min"] for p in out["partes"] if p["tipo"] == "camera")
    out["carros_min"] = sum(p["carros_min"] for p in out["partes"]); out["carros_max"] = None if aberta else sum(p["carros_max"] for p in out["partes"])
    # Nível de precisão (0-100) DO QUE O CARTÃO AFIRMA (não de um total que ele não mostra):
    #  - fila além das câmeras: o cartão só diz "mais de X" e as velocidades vistas -> vale a confiança das câmeras, pesada pelos metros;
    #  - cauda dentro de um trecho cego: o cartão dá uma FAIXA que cobre o trecho inteiro -> 90% do pior vizinho;
    #  - trecho cego dado como livre ou em fila pela regra dos vizinhos: 60% do pior vizinho (é dedução, não medida).
    # É confiança do próprio sistema — ainda NÃO é acerto medido contra travessias cronometradas.
    conf = lambda k: float(t[k].get("confianca") or 0)
    def conf_vao(baixo, cima): return (0.9 if (cheio(baixo) and not fila(cima)) else 0.6) * min(conf(baixo), conf(cima))
    pesos = [(t[k]["trecho_m"], conf(k)) for k in ("ponte", "aduana", "br277")]
    if not aberta: pesos += [(VAO_B, conf_vao("ponte", "aduana")), (VAO_A, conf_vao("aduana", "br277"))]
    prec = sum(m * c for m, c in pesos) / sum(m for m, _ in pesos)
    out["precisao_pct"] = round(prec)
    out.update(estado="estimado" if fmax > fmin or aberta else "medido", fila_min_m=round(fmin), fila_max_m=None if aberta else round(fmax),
               tempo_min_s=round(tmin), tempo_max_s=None if aberta else round(tmax), cauda=cauda,
               fila_local_viaduto=bool(fila("br277") and not cheio("aduana")), percurso_m=VAO_A + VAO_B + sum(t[k]["trecho_m"] for k in ("ponte", "aduana", "br277")))
    if estavel:
        H = _MEM["hist"]; H.append(dict(out)); del H[:-18]
        iguais = [h for h in H if h.get("cauda") == out.get("cauda") and h.get("tempo_min_s") is not None]
        if len(iguais) >= 3:
            med = lambda c: (None if out.get(c) is None else round(float(np.median([h[c] for h in iguais if h.get(c) is not None]))))
            for c in ("fila_min_m", "fila_max_m", "tempo_min_s", "tempo_max_s", "precisao_pct"): out[c] = med(c)
        out["estavel_em"] = len(iguais)
    return out

def leitor(chave, fonte, fila, fim, ao_vivo):
    """ffmpeg -> quadros 1920x1080 BGR, 1 por segundo. Ao vivo guarda só o mais novo (nunca acumula atraso)."""
    cmd = ["ffmpeg", "-loglevel", "error"] + (["-rw_timeout", "15000000"] if ao_vivo else []) + ["-i", fonte, "-an", "-vf", "fps=5,scale=1920:1080", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    while time.time() < fim:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE); n = 0; ant = None
        while time.time() < fim:
            buf = p.stdout.read(1920 * 1080 * 3)
            if len(buf) < 1920 * 1080 * 3: break
            fr = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3); n += 1
            if ao_vivo: fila[chave] = (time.time(), fr, ant)
            elif n % 5 == 0:
                while chave in fila and time.time() < fim: time.sleep(0.01)   # arquivo: espera o quadro ser consumido
                fila[chave] = (n / 5.0, fr, ant)
            ant = fr
        p.kill()
        if not ao_vivo: fila[chave + "_fim"] = True; return
        time.sleep(3)

SURL = "https://aakenoljsycyrcrchgxj.supabase.co"
_tok = [0.0, ""]
def publicar(caminho_local, destino, anon):
    """Envia o JSON para o bucket público camera-clipes/<destino> pela função camera-clipes-subir (token OIDC do GitHub)."""
    try:
        if time.time() - _tok[0] > 180:
            rq = urllib.request.Request(os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] + "&audience=exa-camera-clipes", headers={"Authorization": "bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]})
            _tok[1] = json.load(urllib.request.urlopen(rq, timeout=15))["value"]; _tok[0] = time.time()
        rq = urllib.request.Request(SURL + "/functions/v1/camera-clipes-subir?op=subir&caminho=" + destino, data=open(caminho_local, "rb").read(), method="POST",
                                    headers={"Authorization": "Bearer " + anon, "apikey": anon, "x-gh-oidc": _tok[1], "Content-Type": "application/octet-stream"})
        return urllib.request.urlopen(rq, timeout=20).status
    except Exception as e:
        return "erro: " + str(e)[:120]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modelo", required=True); ap.add_argument("--saida", default="atual.json")
    ap.add_argument("--arquivo", action="append", default=[]); ap.add_argument("--segundos", type=float, default=300); ap.add_argument("--passo", type=float, default=10)
    ap.add_argument("--janela", type=float, default=60); ap.add_argument("--historico", default=None); ap.add_argument("--publicar", default=None, help="caminho no bucket, ex.: fila/atual.json"); a = ap.parse_args()
    anon = os.environ.get("ANON", ""); pub = None
    arqs = dict(x.split("=", 1) for x in a.arquivo); ao_vivo = not arqs
    det = Detector(a.modelo); trechos = {k: Trecho(k, c) for k, c in CAMERAS.items() if ao_vivo or k in arqs}
    fila = {}; fim = time.time() + a.segundos
    for k in trechos: threading.Thread(target=leitor, args=(k, arqs.get(k) or BASE % CAMERAS[k]["stream"], fila, fim, ao_vivo), daemon=True).start()
    prox = None; hist = []; t_ini = time.time(); gasto = []
    while time.time() < fim:
        fez = False
        for k, tr in trechos.items():
            item = fila.pop(k, None) if not ao_vivo else fila.get(k)
            if item is None or (ao_vivo and item[0] <= tr.ultimo): continue
            t0 = time.time(); tr.quadro(item[0], item[1], det, item[2]); gasto.append(time.time() - t0); fez = True
        relogio = time.time() if ao_vivo else max([tr.ultimo for tr in trechos.values()] + [0])
        if prox is None: prox = relogio + a.passo
        if relogio >= prox:
            prox += a.passo
            res = dict(versao="0.16", em=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), t=round(relogio - (t_ini if ao_vivo else 0), 1), sentido="Brasil -> Paraguai",
                       trechos=[tr.resumo(relogio, a.janela) for tr in trechos.values()], ms_por_quadro=round(1000 * float(np.mean(gasto[-30:])) if gasto else 0))
            res["travessia"] = travessia(res["trechos"])
            json.dump(res, open(a.saida + ".tmp", "w"), ensure_ascii=False); os.replace(a.saida + ".tmp", a.saida); hist.append(res); hist[:] = hist[-360:]
            if a.publicar: pub = publicar(a.saida, a.publicar, anon)
            if a.publicar or a.historico:      # para a página de monitor: histórico da última hora + o quadro que o sistema acabou de analisar em cada câmera
                def curto(r):
                    t = {x["trecho"]: x for x in r["trechos"]}; v = r.get("travessia") or {}
                    return [r["em"], v.get("tempo_min_s"), v.get("tempo_max_s"), v.get("fila_min_m"), v.get("precisao_pct")] + [t.get(k, {}).get(c) for k in ("ponte", "aduana", "br277") for c in ("kmh", "fila_m", "veiculos")]
                json.dump(dict(campos=["em", "tempo_min_s", "tempo_max_s", "fila_min_m", "precisao_pct", "ponte_kmh", "ponte_fila_m", "ponte_veic", "aduana_kmh", "aduana_fila_m", "aduana_veic", "br277_kmh", "br277_fila_m", "br277_veic"],
                               pontos=[curto(r) for r in hist[::3]], em=res["em"], quadros={k: getattr(tr, "ult", None) for k, tr in trechos.items()}), open(a.saida + ".hist", "w"))
                if a.publicar: publicar(a.saida + ".hist", a.publicar.replace("atual.json", "historico.json"), anon)
            print(json.dumps({"t": res["t"], **{x["trecho"]: [x["estado"], x["veiculos"], x["parado_pct"], x["kmh"], x["fila_m"]] for x in res["trechos"]}, "ms": res["ms_por_quadro"], "pub": pub}, ensure_ascii=False), flush=True)
        if not ao_vivo and all((k + "_fim") in fila and k not in fila for k in trechos): break
        if not fez: time.sleep(0.02)
    if a.historico: json.dump(hist, open(a.historico, "w"), ensure_ascii=False)

if __name__ == "__main__": main()
