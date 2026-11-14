"""Calibración intrínseca de la cámara con tablero de ajedrez (OpenCV).
Genera calibracion_camara.npz con la matriz K y los coeficientes de distorsión.
Uso: s = guardar vista (cuando se detectan las esquinas) | c = calibrar | q = salir
Tomar 15-25 vistas con el tablero en distintas posiciones, distancias e inclinaciones."""
import cv2
import numpy as np

CAM_INDEX = 0
PATRON = (9, 6)        # esquinas INTERNAS del tablero (columnas, filas)
CUADRO_MM = 25.0       # lado real de cada cuadro, medido con regla

objp = np.zeros((PATRON[0] * PATRON[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATRON[0], 0:PATRON[1]].T.reshape(-1, 2) * CUADRO_MM
pts_obj, pts_img, tam = [], [], None
cap = cv2.VideoCapture(CAM_INDEX)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tam = gris.shape[::-1]
    hallado, esq = cv2.findChessboardCorners(gris, PATRON)
    vista = frame.copy()
    if hallado:
        esq = cv2.cornerSubPix(gris, esq, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(vista, PATRON, esq, hallado)
    cv2.putText(vista, f"vistas: {len(pts_img)}  (s guardar, c calibrar, q salir)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Calibracion", vista)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s') and hallado:
        pts_obj.append(objp)
        pts_img.append(esq)
    elif k == ord('c') and len(pts_img) >= 10:
        rms, K, dist, _, _ = cv2.calibrateCamera(pts_obj, pts_img, tam, None, None)
        np.savez("calibracion_camara.npz", K=K, dist=dist, rms=rms, tam=tam)
        print(f"Error RMS de reproyección: {rms:.3f} px (idealmente < ~0.5)")
        print("K =\n", K, "\ndist =", dist.ravel())
        break
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
