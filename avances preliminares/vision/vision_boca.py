"""Localización 3D del punto medio de los labios (banco de pruebas, OE4).
Cámara RGB (eye-in-hand) + sensor ToF VL53L0X leído por un ESP32 vía serial.
Requisitos: pip install opencv-python mediapipe numpy pyserial
(si tu versión de mediapipe ya no incluye mp.solutions, usa la API Face Landmarker: mismos índices)
Salida: ventana con la boca localizada, CSV con cada medición y estadísticas al cerrar (q)."""
import csv
import json
import os
import socket
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import serial

# ----------------------------- Configuración -----------------------------
CAM_INDEX = 0
PUERTO_TOF, BAUD = "COM3", 115200              # Linux: "/dev/ttyUSB0"
CALIB_FILE = "calibracion_camara.npz"          # generado por calibrar_camara.py
T_EE_CAM_FILE = "T_ee_cam.npy"                 # calibración mano-ojo (PENDIENTE: identidad por ahora)
TOF_OFFSET_MM = 0.0     # el ToF ve lo más cercano (nariz/mentón): calibrar este desfase midiendo con regla
TOF_MIN_MM, TOF_MAX_MM, TOF_FRESCO_S = 30, 1200, 0.2
ANCHO_BOCA_MM = (35, 75)                        # verificación de coherencia (ancho real típico ~50 mm)
PESOS = {13: 0.35, 14: 0.35, 61: 0.15, 291: 0.15}   # labio sup./inf. interno y comisuras (suman 1)
ALPHA = 0.4                                     # suavizado exponencial (0..1; menor = más suave)
TIMEOUT_ROSTRO_S = 0.5                          # pérdida de detección > 500 ms -> PARADA (OE6)
LOG_FILE = "mediciones_boca.csv"
ENVIAR_SIM = False                              # enviar la medición por UDP a un brazo virtual (receptor no incluido en esta entrega)
SIM_ADDR = ("127.0.0.1", 5005)

# --------------------------- Calibración y poses ---------------------------
calib = np.load(CALIB_FILE)
K, DIST = calib["K"], calib["dist"]
T_EE_CAM = np.load(T_EE_CAM_FILE) if os.path.exists(T_EE_CAM_FILE) else np.eye(4)
T_BASE_EE = np.eye(4)   # pendiente: recibir del controlador (cinemática directa con los ángulos actuales)


def normalizado(u, v):
    """Pixel -> coordenadas normalizadas sin distorsión (x/z, y/z) usando K y dist."""
    p = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), K, DIST)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


# ------------------------------ Lectura del ToF ------------------------------
class LectorToF(threading.Thread):
    """Lee líneas con la distancia en mm que envía el ESP32 y guarda la última."""
    def __init__(self):
        super().__init__(daemon=True)
        self.ser = serial.Serial(PUERTO_TOF, BAUD, timeout=0.05)
        self.mm, self.t = None, 0.0
        self.activo = True

    def run(self):
        while self.activo:
            linea = self.ser.readline().decode(errors="ignore").strip()
            if linea.isdigit():
                self.mm, self.t = int(linea), time.time()

    def distancia(self):
        if self.mm is None or time.time() - self.t > TOF_FRESCO_S:
            return None
        return self.mm if TOF_MIN_MM <= self.mm <= TOF_MAX_MM else None


# ------------------------------ Programa principal ------------------------------
mp_fm = mp.solutions.face_mesh
LABIOS = sorted({i for par in mp_fm.FACEMESH_LIPS for i in par})
tof = LectorToF()
tof.start()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(CAM_INDEX)
suave_uv, suave_z = None, None
t_ultimo_rostro = time.time()
tiempos_ms, n_validas = [], 0

with open(LOG_FILE, "w", newline="") as f, mp_fm.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    log = csv.writer(f)
    log.writerow(["t_s", "u_px", "v_px", "tof_mm", "X_mm", "Y_mm", "Z_mm",
                  "ancho_boca_mm", "valida", "Xr_mm", "Yr_mm", "Zr_mm", "ms_proc"])
    t_inicio = time.time()
    while cap.isOpened():
        ok, img = cap.read()                      # SIN voltear: el espejo invertiría el eje X
        if not ok:
            print("No se pudo acceder a la cámara.")
            break
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        estado = "SIN ROSTRO"
        enviado = False

        if res.multi_face_landmarks:
            t_ultimo_rostro = time.time()
            lm = res.multi_face_landmarks[0].landmark
            # 1) Centro de la boca: promedio ponderado de varios landmarks + suavizado temporal
            u = sum(p * lm[i].x * w for i, p in PESOS.items())
            v = sum(p * lm[i].y * h for i, p in PESOS.items())
            suave_uv = (u, v) if suave_uv is None else (ALPHA * u + (1 - ALPHA) * suave_uv[0],
                                                        ALPHA * v + (1 - ALPHA) * suave_uv[1])
            # 2) Caja delimitadora de la boca (región de interés)
            xs = [lm[i].x * w for i in LABIOS]
            ys = [lm[i].y * h for i in LABIOS]
            cv2.rectangle(img, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))), (0, 255, 0), 2)
            cv2.circle(img, (int(suave_uv[0]), int(suave_uv[1])), 5, (255, 0, 0), -1)

            # 3) Fusión con el ToF: píxel + distancia -> (X, Y, Z) en mm, sistema de la cámara
            tof_mm = tof.distancia()
            if tof_mm is None:
                estado = "SIN DISTANCIA ToF"
                ms = (time.perf_counter() - t0) * 1000
                tiempos_ms.append(ms)
            else:
                z = tof_mm + TOF_OFFSET_MM
                suave_z = z if suave_z is None else ALPHA * z + (1 - ALPHA) * suave_z
                xn, yn = normalizado(*suave_uv)                    # X = xn*Z, Y = yn*Z (modelo pinhole)
                X, Y, Z = xn * suave_z, yn * suave_z, suave_z
                # 4) Verificación: el ancho de la boca en mm debe ser coherente con la distancia
                x1, y1 = normalizado(lm[61].x * w, lm[61].y * h)
                x2, y2 = normalizado(lm[291].x * w, lm[291].y * h)
                ancho = float(np.hypot(x1 - x2, y1 - y2) * suave_z)
                valida = ANCHO_BOCA_MM[0] <= ancho <= ANCHO_BOCA_MM[1]
                # 5) Al sistema del robot: p_robot = T_base_ee * T_ee_cam * p_cam (calibración pendiente)
                Xr, Yr, Zr = (T_BASE_EE @ T_EE_CAM @ np.array([X, Y, Z, 1.0]))[:3]
                ms = (time.perf_counter() - t0) * 1000
                tiempos_ms.append(ms)
                n_validas += int(valida)
                estado = "OK" if valida else "DETECCION DUDOSA"
                log.writerow([f"{time.time() - t_inicio:.3f}", f"{suave_uv[0]:.1f}", f"{suave_uv[1]:.1f}",
                              tof_mm, f"{X:.1f}", f"{Y:.1f}", f"{Z:.1f}", f"{ancho:.1f}", int(valida),
                              f"{Xr:.1f}", f"{Yr:.1f}", f"{Zr:.1f}", f"{ms:.1f}"])
                if ENVIAR_SIM:   # coordenadas en el marco de la cámara (OpenCV), en mm
                    sock.sendto(json.dumps({"t": time.time(), "X": X, "Y": Y, "Z": Z, "valida": bool(valida)}).encode(), SIM_ADDR)
                    enviado = True
                cv2.putText(img, f"X:{X:.0f} Y:{Y:.0f} Z:{Z:.0f} mm  ancho:{ancho:.0f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif time.time() - t_ultimo_rostro > TIMEOUT_ROSTRO_S:
            estado = "PARADA: perdida de deteccion"   # aquí se enviaría la orden de parada al ESP32
            suave_uv = suave_z = None

        if ENVIAR_SIM and not enviado:    # latido sin detección: distingue 'sin rostro' de 'sin comunicación'
            sock.sendto(json.dumps({"t": time.time(), "valida": False}).encode(), SIM_ADDR)
        cv2.putText(img, f"Estado: {estado}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if estado != "OK" else (0, 200, 0), 2)
        cv2.imshow("Banco de vision - boca XYZ", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

tof.activo = False
cap.release()
cv2.destroyAllWindows()
if tiempos_ms:
    t = np.array(tiempos_ms)
    print(f"Fotogramas: {len(t)} | válidas: {n_validas} | tiempo medio: {t.mean():.1f} ms "
          f"| p95: {np.percentile(t, 95):.1f} ms | FPS ~ {1000 / t.mean():.0f}")
