#!/usr/bin/env python3
"""EXA — fila da Ponte da Amizade (Brasil -> Paraguai) em tempo real. v0.4 (12/11/2026)

Lê as câmeras fixas do Portal da Cidade (ou gravações, para teste), detecta cada veículo (YOLOX-s, Apache 2.0,
via OpenCV), acompanha veículo a veículo e publica a cada PASSO segundos um resumo por trecho:
veículos, parados, estado (livre / lento / parado), velocidade e metros de fila (onde a câmera já tem escala) e confiança.
Regra do Jeferson: quando não dá para ver direito, NÃO publica número — o trecho sai como "indisponivel".

uso:  fila_worker.py --modelo yolox_s.onnx --saida atual.json [--arquivo cam=video.ts ...] [--segundos 300] [--passo 10]
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request
import cv2, numpy as np

BASE = "https://video02.logicahost.com.br/portaldacidade/%s.stream/playlist.m3u8"
VEIC = {2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}
PAR = 0.2   # segundos entre os dois quadros do par (5 quadros/s)

# Cada câmera fixa = um trecho. Coordenadas na imagem 1920x1080.
#  recortes: onde o detector olha (640x640)   eixo: sentido de quem vai para o Paraguai (na imagem)
#  zona: polígono das faixas de CARRO sentido Paraguai (caminhão em fila própria fica fora)
#  metros: tabela y -> metros até o fim do trecho (None = câmera ainda sem escala medida)
CAMERAS = {
    "ponte": dict(stream="fozpontedaamizadesentidoparaguai", nome="Ponte (ponta paraguaia)",
        recortes=[(450, 100, 1090, 740), (700, 440, 1340, 1080), (1000, 440, 1640, 1080)],
        eixo=(-0.58, -0.81),
        zona=[(700, 250), (790, 250), (1010, 600), (1420, 1080), (930, 1080), (880, 760), (760, 480)],
        # escala provisória (±30%): perfil de tempo de travessia de 12/11 ancorado nos 132 m mirante -> cabeceira
        metros=[(202, 0), (255, 10), (330, 34), (450, 64), (600, 92), (742, 113), (900, 125), (1050, 134)]),
    "aduana": dict(stream="fozaduanapontedaamizade", nome="Curva da aduana",
        recortes=[(560, 0, 1200, 640), (200, 280, 840, 920), (700, 280, 1340, 920)],
        eixo=(-0.80, 0.60), ignora=(7,),
        zona=[(1075, 20), (1215, 20), (1150, 300), (900, 560), (420, 900), (60, 900), (640, 470), (1000, 250)],
        # escala provisória (±25%): comprimento dos carros na imagem (perto ~20 px/m, fundo ~7,6 px/m ao longo da pista)
        metros=[(20, 145), (64, 127), (117, 107), (200, 86), (313, 67), (387, 54), (493, 40), (640, 18), (800, 6), (900, 0)]),
    "br277": dict(stream="fozsentidopontedaamizade01", nome="BR-277 / viaduto",
        recortes=[(560, 300, 1200, 940), (640, 560, 1280, 1080)],
        eixo=(-0.05, -1.0), ignora=(7,),
        zona=[(858, 420), (990, 420), (1012, 1080), (868, 1080)],
        # escala provisória (±15%): ~3,4 px/m ao longo da pista (faixas de 3,5 m e carretas de 18,6 m como régua);
        # confere com o mapa: o trecho visível são os ~195 m finais da reta antes do viaduto (nó do OSM em -25.50601,-54.58833)
        metros=[(420, 0), (500, 45), (600, 98), (700, 148), (800, 196), (900, 240), (1000, 282), (1080, 314)]),
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
    def quadro(s, agora, img, det, img_ant=None):
        b, c, k = det.detectar(img, s.cfg["recortes"]); vivos = s.r.passo(agora, b, k); s.quadros += 1; s.ultimo = agora
        cinza = cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY); s.brilho.append(float(cinza.mean())); s.nitidez.append(float(cv2.Laplacian(cinza, cv2.CV_64F).var()))
        na_zona = 0
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g0 = cv2.cvtColor(img_ant, cv2.COLOR_BGR2GRAY) if img_ant is not None else None
        for i, v in vivos.items():
            if v["visto"] != agora or v["k"] == 3 or v["k"] in s.cfg.get("ignora", ()): continue                      # moto não entra na fila de carro
            x, y = v["h"][-1][1], v["h"][-1][2]
            if cv2.pointPolygonTest(s.zona, (x, y), False) < 0: continue
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
    def resumo(s, agora, janela):
        A = [a for a in s.amostras if a["t"] >= agora - janela]; s.amostras = [a for a in s.amostras if a["t"] >= agora - 120]
        out = dict(trecho=s.chave, nome=s.cfg["nome"], trecho_m=(max(m for _, m in s.cfg["metros"]) if s.cfg["metros"] else None), estado="indisponivel", motivo=None, veiculos=None, parados=None, parado_pct=None, kmh=None, fila_m=None, tempo_trecho_s=None, confianca=0)
        if agora - s.ultimo > 15 or s.quadros < 3: out["motivo"] = "camera sem imagem"; return out
        br, nt = np.mean(s.brilho[-8:]), np.mean(s.nitidez[-8:])
        if br < 35: out["motivo"] = "escuro demais"; return out
        if nt < 25: out["motivo"] = "imagem sem nitidez (chuva, neblina ou camera fora de foco)"; return out
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
def travessia(tr):
    t = {x["trecho"]: x for x in tr}
    out = dict(estado="indisponivel", motivo=None, fila_min_m=None, fila_max_m=None, tempo_min_s=None, tempo_max_s=None, cauda=None)
    if any(k not in t or t[k]["estado"] == "indisponivel" for k in ("ponte", "aduana", "br277")):
        out["motivo"] = "trecho sem leitura: " + ", ".join(k for k in ("ponte", "aduana", "br277") if k not in t or t[k]["estado"] == "indisponivel"); return out
    fila = lambda k: t[k]["estado"] in ("lento", "parado")
    cheio = lambda k: fila(k) and (t[k]["fila_m"] or 0) >= 0.7 * t[k]["trecho_m"]      # cauda passa do que a câmera vê
    vel = lambda k: max((t[k]["kmh"] or 0) / 3.6, 0.3) if fila(k) else max((t[k]["kmh"] or 0) / 3.6, V_LIVRE)
    tmin = tmax = sum(t[k]["trecho_m"] / vel(k) for k in ("ponte", "aduana", "br277"))
    fmin = fmax = 0; cauda = "sem fila na ponta paraguaia"; aberta = False
    def vao(comp, baixo, cima):
        # devolve (tempo_min, tempo_max, fila_min, fila_max, continua_para_cima)
        if not cheio(baixo): return comp / V_LIVRE, comp / V_LIVRE, 0, 0, False
        if fila(cima):                                    # fila nas duas pontas -> o meio também está em fila
            a, b = sorted((vel(baixo), vel(cima))); return comp / b, comp / a, comp, comp, True
        return comp / V_LIVRE, comp / vel(baixo), 0, comp, False   # cauda está dentro do trecho cego
    if fila("ponte"):
        fmin = fmax = t["ponte"]["fila_m"] or 0; cauda = "na ponte (visivel)"
        a, b, c, d, segue = vao(VAO_B, "ponte", "aduana"); tmin += a; tmax += b; fmin += c; fmax += d
        if cheio("ponte"): cauda = "entre a aduana e a ponte (sem camera)"
        if segue:
            fmin += t["aduana"]["fila_m"] or 0; fmax += t["aduana"]["fila_m"] or 0; cauda = "na curva da aduana (visivel)"
            a, b, c, d, segue = vao(VAO_A, "aduana", "br277"); tmin += a; tmax += b; fmin += c; fmax += d
            if cheio("aduana"): cauda = "na curva antes da aduana (sem camera)"
            if segue:
                fmin += t["br277"]["fila_m"] or 0; fmax += t["br277"]["fila_m"] or 0; cauda = "na BR-277 (visivel)"
                if cheio("br277"): aberta = True; cauda = "antes do viaduto (alem das cameras)"
        else: tmin += VAO_A / V_LIVRE; tmax += VAO_A / V_LIVRE
    else:
        tmin += (VAO_A + VAO_B) / V_LIVRE; tmax += (VAO_A + VAO_B) / V_LIVRE
    out.update(estado="estimado" if fmax > fmin or aberta else "medido", fila_min_m=round(fmin), fila_max_m=None if aberta else round(fmax),
               tempo_min_s=round(tmin), tempo_max_s=None if aberta else round(tmax), cauda=cauda,
               fila_local_viaduto=bool(fila("br277") and not cheio("aduana")), percurso_m=VAO_A + VAO_B + sum(t[k]["trecho_m"] for k in ("ponte", "aduana", "br277")))
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
            res = dict(versao="0.5", em=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), t=round(relogio - (t_ini if ao_vivo else 0), 1), sentido="Brasil -> Paraguai",
                       trechos=[tr.resumo(relogio, a.janela) for tr in trechos.values()], ms_por_quadro=round(1000 * float(np.mean(gasto[-30:])) if gasto else 0))
            res["travessia"] = travessia(res["trechos"])
            json.dump(res, open(a.saida + ".tmp", "w"), ensure_ascii=False); os.replace(a.saida + ".tmp", a.saida); hist.append(res); hist[:] = hist[-360:]
            if a.publicar: pub = publicar(a.saida, a.publicar, anon)
            print(json.dumps({"t": res["t"], **{x["trecho"]: [x["estado"], x["veiculos"], x["parado_pct"], x["kmh"], x["fila_m"]] for x in res["trechos"]}, "ms": res["ms_por_quadro"], "pub": pub}, ensure_ascii=False), flush=True)
        if not ao_vivo and all((k + "_fim") in fila and k not in fila for k in trechos): break
        if not fez: time.sleep(0.02)
    if a.historico: json.dump(hist, open(a.historico, "w"), ensure_ascii=False)

if __name__ == "__main__": main()
