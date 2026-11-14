"""Geometría pura (sin cámara ni MediaPipe) para pasar de píxeles de la webcam a la posición de la boca en el marco del robot.
Modelo: cámara fija con modelo pinhole; ancho real de la boca supuesto (comisura a comisura)."""
import numpy as np

ANCHO_BOCA_MM = 50.0          # SUPUESTO: ancho típico comisura-comisura; varía entre personas (medir al usuario real)


def focal_px(ancho_img_px, hfov_deg=65.0):
    """Distancia focal en píxeles a partir del campo de visión horizontal (típico de webcams: 60-70°). Calibrar con calibrar_camara.py."""
    return ancho_img_px / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))


def profundidad_por_ancho(ancho_px, f_px, ancho_mm=ANCHO_BOCA_MM):
    """Z = f * ancho_real / ancho_en_pixeles (aprox.: cae si el usuario gira la cabeza)."""
    return f_px * ancho_mm / max(ancho_px, 1e-6)


def pixel_a_cam(u, v, z, f_px, cx, cy):
    """Pixel + profundidad -> (X, Y, Z) [mm] en el marco de la cámara (OpenCV: x derecha, y abajo, z adelante)."""
    return np.array([(u - cx) * z / f_px, (v - cy) * z / f_px, z])


# Cámara junto al brazo mirando al usuario (mismo sentido que +x del robot, z arriba):
#   x_cámara (derecha de la imagen) -> -y_robot ; y_cámara (abajo) -> -z_robot ; z_cámara (adelante) -> +x_robot
R_CAM_A_ROBOT = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])


def cam_a_robot(p_cam, p_ref_cam, boca_nominal, semi_volumen=None):
    """Desplazamiento respecto de la referencia (tecla 'c') aplicado a la boca nominal del robot; se limita al volumen acotado."""
    p = np.asarray(boca_nominal, float) + R_CAM_A_ROBOT @ (np.asarray(p_cam, float) - np.asarray(p_ref_cam, float))
    if semi_volumen is not None:
        p = np.clip(p, np.asarray(boca_nominal) - semi_volumen, np.asarray(boca_nominal) + semi_volumen)
    return p


def boca_abierta(lm, w, h, umbral=0.25):
    """Relación apertura/ancho de la boca con los puntos internos 13 (labio sup.), 14 (inf.), 61 y 291 (comisuras)."""
    ap = np.hypot((lm[13].x - lm[14].x) * w, (lm[13].y - lm[14].y) * h)
    an = np.hypot((lm[61].x - lm[291].x) * w, (lm[61].y - lm[291].y) * h)
    r = ap / max(an, 1e-6)
    return bool(r > umbral), float(r)
