#!/usr/bin/env python3
"""Diagnóstico: pega quadros das câmeras agora, roda o detector como o leitor roda e desenha zona, recortes e caixas."""
import sys, os, json, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fila"))
import cv2, numpy as np
import fila_worker as W
modelo, saida = sys.argv[1], sys.argv[2]; os.makedirs(saida, exist_ok=True)
det = W.Detector(modelo); res = {}
EXTRA = {"giratoria": "fozpontedaamizadesentidobrasil"}
def pega(stream, n=3, passo=4):
    out = []
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-rw_timeout", "15000000", "-i", W.BASE % stream, "-an", "-vf", "fps=1/%d,scale=1920:1080" % passo,
                        "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, timeout=90)
    b = p.stdout
    for i in range(len(b) // (1920 * 1080 * 3)): out.append(np.frombuffer(b[i * 6220800:(i + 1) * 6220800], np.uint8).reshape(1080, 1920, 3).copy())
    return out
for k, cfg in W.CAMERAS.items():
    qs = pega(cfg["stream"]); res[k] = []
    for i, img in enumerate(qs):
        cv2.imwrite("%s/%s_%d_bruto.jpg" % (saida, k, i), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b, c, kl = det.detectar(img, cfg["recortes"]); zona = np.array(cfg["zona"], np.int32); d = img.copy()
        for (x0, y0, x1, y1) in cfg["recortes"]: cv2.rectangle(d, (x0, y0), (x1, y1), (255, 120, 0), 1)
        cv2.polylines(d, [zona], True, (0, 255, 255), 2)
        cont = dict(detectados=len(b), na_zona=0, por_classe={})
        for bb, cc, kk in zip(b, c, kl):
            x, y = (bb[0] + bb[2]) / 2, bb[3]; dentro = cv2.pointPolygonTest(zona, (float(x), float(y)), False) >= 0
            conta = dentro and int(kk) != 3 and int(kk) not in cfg.get("ignora", ())
            cont["na_zona"] += int(conta); n = W.VEIC[int(kk)]; cont["por_classe"][n] = cont["por_classe"].get(n, 0) + 1
            cv2.rectangle(d, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0) if conta else (0, 0, 255), 2)
        # detector com lupa: recortes de 320 px ampliados 2x sobre toda a área dos recortes atuais (o que um ajuste melhor acharia)
        xs0 = min(r[0] for r in cfg["recortes"]); ys0 = min(r[1] for r in cfg["recortes"]); xs1 = max(r[2] for r in cfg["recortes"]); ys1 = max(r[3] for r in cfg["recortes"])
        g = cv2.resize(img[ys0:ys1, xs0:xs1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); H, Wd = g.shape[:2]
        rec2 = [(x, y, min(x + 640, Wd), min(y + 640, H)) for y in range(0, max(H - 320, 1), 480) for x in range(0, max(Wd - 320, 1), 480)]
        b2, c2, k2 = det.detectar(g, rec2, 0.25); d2 = img.copy(); cv2.polylines(d2, [zona], True, (0, 255, 255), 2); n2 = 0
        for bb, kk in zip(b2, k2):
            bb = bb / 2 + np.array([xs0, ys0, xs0, ys0]); x, y = (bb[0] + bb[2]) / 2, bb[3]
            dentro = cv2.pointPolygonTest(zona, (float(x), float(y)), False) >= 0; n2 += int(dentro and int(kk) != 3)
            cv2.rectangle(d2, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0) if dentro else (0, 0, 255), 2)
        cont["lupa_detectados"] = len(b2); cont["lupa_na_zona_sem_moto"] = n2; res[k].append(cont)
        cv2.imwrite("%s/%s_%d_det.jpg" % (saida, k, i), d, [cv2.IMWRITE_JPEG_QUALITY, 88]); cv2.imwrite("%s/%s_%d_lupa.jpg" % (saida, k, i), d2, [cv2.IMWRITE_JPEG_QUALITY, 88])
for k, st in EXTRA.items():
    for i, img in enumerate(pega(st, 6, 12)): cv2.imwrite("%s/%s_%d_bruto.jpg" % (saida, k, i), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
res["em"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); json.dump(res, open(saida + "/contagem.json", "w"), indent=1); print(json.dumps(res))
