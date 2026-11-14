"""Detección de la boca con la webcam del computador y envío de su posición al brazo virtual.
Basado en el script de Face Mesh de Ignacio. Qué añade:
  - dibuja solo los labios (más rápido y más claro) + caja de la boca + punto ponderado (landmarks 13, 14, 61, 291)
  - indica si la boca está abierta o cerrada
  - estima la posición 3D SIN sensor ToF: la profundidad sale del ancho de la boca en píxeles (aprox., ver geom_boca.py)
  - NO voltea la imagen (el espejo invierte el eje x y los cálculos)
  - 'c' fija la posición actual como la boca nominal del robot; 'q' sale
  - envía por UDP (127.0.0.1:5005) {t, valida, boca_robot=[x,y,z] mm, abierta}; si pierde el rostro envía solo el latido
Uso:  python3 boca_webcam.py [--cam 0] [--hfov 65] [--sin-envio]
Requisitos: pip install opencv-python mediapipe numpy"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOCA_NOMINAL, VOLUMEN_BOCA
from geom_boca import *

import mediapipe as mp

# Contornos de los labios (índices de Face Mesh): exterior e interior, como polilíneas cerradas
LABIO_EXT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LABIO_INT = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LABIOS = sorted(set(LABIO_EXT + LABIO_INT))
MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
URL_MODELO = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def crear_detector():
    """Devuelve (detectar(img_bgr) -> lista de 468+ landmarks o None, cerrar()).
    Usa la API antigua (mp.solutions) si existe; si no, la nueva (FaceLandmarker, descarga el modelo la primera vez)."""
    if hasattr(mp, "solutions"):
        fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
        def detectar(img):
            r = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return r.multi_face_landmarks[0].landmark if r.multi_face_landmarks else None
        return detectar, fm.close
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision
    if not os.path.exists(MODELO):
        import urllib.request
        print("Descargando el modelo face_landmarker.task (una sola vez)...")
        urllib.request.urlretrieve(URL_MODELO, MODELO)
    op = vision.FaceLandmarkerOptions(base_options=mp_py.BaseOptions(model_asset_path=MODELO),
                                      running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                      min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    fl = vision.FaceLandmarker.create_from_options(op)
    t0 = time.time()
    def detectar(img):
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        r = fl.detect_for_video(mpi, int((time.time() - t0) * 1000) + 1)
        return r.face_landmarks[0] if r.face_landmarks else None
    return detectar, fl.close


PESOS = {13: 0.35, 14: 0.35, 61: 0.15, 291: 0.15}     # igual que vision_boca.py (labios internos y comisuras)
ALPHA = 0.4                                            # suavizado exponencial
SIM_ADDR = ("127.0.0.1", 5005)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--hfov", type=float, default=65.0, help="campo de visión horizontal de la webcam [°]")
    ap.add_argument("--sin-envio", action="store_true")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(a.cam)
    detectar, cerrar = crear_detector()
    suave_uv, suave_z, ref = None, None, None
    if True:
        while cap.isOpened():
            ok, img = cap.read()
            if not ok:
                print("No se pudo acceder a la cámara.")
                break
            h, w = img.shape[:2]
            f_px, cx, cy = focal_px(w, a.hfov), w / 2.0, h / 2.0
            lm = detectar(img)
            estado, enviado = "SIN ROSTRO", False

            if lm is not None:
                for anillo in (LABIO_EXT, LABIO_INT):                       # contorno de los labios
                    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in anillo], np.int32)
                    cv2.polylines(img, [pts], True, (0, 255, 0), 1)
                u = sum(p * lm[i].x * w for i, p in PESOS.items())
                v = sum(p * lm[i].y * h for i, p in PESOS.items())
                suave_uv = (u, v) if suave_uv is None else (ALPHA * u + (1 - ALPHA) * suave_uv[0], ALPHA * v + (1 - ALPHA) * suave_uv[1])
                xs = [lm[i].x * w for i in LABIOS]
                ys = [lm[i].y * h for i in LABIOS]
                cv2.rectangle(img, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))), (0, 255, 0), 2)
                cv2.circle(img, (int(suave_uv[0]), int(suave_uv[1])), 5, (255, 0, 0), -1)

                ancho_px = float(np.hypot((lm[61].x - lm[291].x) * w, (lm[61].y - lm[291].y) * h))
                z = profundidad_por_ancho(ancho_px, f_px)
                suave_z = z if suave_z is None else ALPHA * z + (1 - ALPHA) * suave_z
                p_cam = pixel_a_cam(*suave_uv, suave_z, f_px, cx, cy)
                abierta, razon = boca_abierta(lm, w, h)
                cv2.putText(img, f"Z~{suave_z:.0f} mm  boca {'ABIERTA' if abierta else 'cerrada'} ({razon:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if ref is None:
                    estado = "pulsa 'c' con la boca en su posicion nominal"
                else:
                    p_rob = cam_a_robot(p_cam, ref, BOCA_NOMINAL, VOLUMEN_BOCA)
                    estado = "OK"
                    cv2.putText(img, f"robot x:{p_rob[0]:.0f} y:{p_rob[1]:.0f} z:{p_rob[2]:.0f} mm", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    if not a.sin_envio:
                        msg = dict(t=time.time(), valida=True, boca_robot=[float(x) for x in p_rob], abierta=abierta)
                        sock.sendto(json.dumps(msg).encode(), SIM_ADDR)
                        enviado = True
            else:
                suave_uv = suave_z = None

            if not a.sin_envio and not enviado:        # latido sin detección: distingue 'sin rostro' de 'sin comunicación'
                sock.sendto(json.dumps(dict(t=time.time(), valida=False)).encode(), SIM_ADDR)
            cv2.putText(img, f"Estado: {estado}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if estado == "OK" else (0, 0, 255), 2)
            cv2.imshow("Boca -> brazo virtual   (c = fijar referencia, q = salir)", img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c") and suave_uv is not None:
                ref = p_cam.copy()
                print("Referencia fijada:", np.round(ref, 1), "mm (marco de la cámara)")
    cerrar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
