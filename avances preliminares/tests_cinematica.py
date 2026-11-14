"""Pruebas de verificación del modelo (ejecutar: python tests_cinematica.py  |  o: pytest)."""
import numpy as np
from config import *
from cinematica import *
from trayectoria import *


def _q_aleatorios(n, seed=1):
    return np.random.default_rng(seed).uniform(Q_MIN, Q_MAX, (n, 5))


def test_fk_extendido():
    assert np.allclose(punta(np.zeros(5)), [L2 + L3 + L4, 0, H])


def test_fk_matricial_vs_cerrada():
    Q = _q_aleatorios(500)
    P, b = fk_lote(Q)
    for q, p, bb in zip(Q, P, b):
        assert np.linalg.norm(punta(q) - p) < 1e-9 and abs(beta_herramienta(q) - bb) < 1e-12


def test_ik_ida_vuelta():
    for q in _q_aleatorios(2000):
        for codo in (-1, 1):
            qq = ik_analitica(punta(q), beta_herramienta(q), codo, q[4])
            if qq is not None:
                assert np.linalg.norm(punta(qq) - punta(q)) < 1e-8
                assert abs(beta_herramienta(qq) - beta_herramienta(q)) < 1e-10


def test_ik_numerica_coincide():
    for q in _q_aleatorios(100, seed=3):
        p, b = punta(q), beta_herramienta(q)
        qn = ik_numerica(p, b, q0=q + 0.05)   # arranque cercano: cerca de una singularidad el método numérico es lento
        assert np.linalg.norm(punta(qn) - p) < 1e-3


def test_jacobiano_diferencias_finitas():
    for q in _q_aleatorios(50, seed=4):
        J = jacobiano_pos(q)
        Jn = np.zeros((3, 5))
        for i in range(5):
            dq = np.zeros(5); dq[i] = 1e-6
            Jn[:, i] = (punta(q + dq) - punta(q - dq)) / 2e-6
        assert np.abs(J - Jn).max() < 1e-4


def test_pose_entrega_distancia():
    m = BOCA_NOMINAL + [10, -20, 5]
    assert abs(np.linalg.norm(pose_entrega(m, STANDOFF_ENTREGA) - m) - STANDOFF_ENTREGA) < 1e-9


def test_trayectoria_velocidades():
    q = Q_HOME.copy()
    for nombre, p0, p1, beta, vmax in plan_transporte(q, BOCA_NOMINAL):
        P, Q = segmento_cartesiano(p0, p1, beta, vmax, q)
        assert P is not None, nombre
        v = np.linalg.norm(np.diff(np.vstack([p0, P]), axis=0), axis=1) / T_CTRL
        assert v.max() <= vmax * 1.01, (nombre, v.max())
        assert np.abs(np.diff(np.vstack([q, Q]), axis=0)).max() / T_CTRL <= VEL_ART_MAX, nombre
        assert all(en_limites(x) for x in Q)
        q = Q[-1]


def test_marco_camara():
    """Un punto 300 mm delante de la cuchara debe verse al centro (x≈0), 30 mm bajo la cámara (y≈+30) y a z≈350."""
    q = Q_HOME.copy()
    T = fk(q) @ T_EE_CAM
    p_mundo = punta(q) + 300 * np.array([1, 0, 0])
    p_cam = (np.linalg.inv(T) @ np.append(p_mundo, 1))[:3]
    assert np.allclose(p_cam, [0, 30, 350], atol=1e-6), p_cam


if __name__ == "__main__":
    n = 0
    for nombre, f in sorted(globals().items()):
        if nombre.startswith("test_"):
            f(); n += 1; print("OK ", nombre)
    print(f"{n} pruebas superadas")
