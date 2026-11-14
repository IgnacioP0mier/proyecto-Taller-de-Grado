"""Pruebas simuladas de la lógica de seguridad (OE6): se inyecta cada condición de riesgo y se mide
cuánto tarda el monitor en pasar a PARADA y si mantiene la posición.
Mide la latencia LÓGICA del monitor (periodo de control = T_CTRL). La latencia real incluye actuadores
y comunicación y debe medirse en el hardware."""
import csv
import json
import os
import numpy as np
from config import *
from cinematica import *
from seguridad import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
T_FALLA = 1.0


def correr(nombre, aplicar, umbral):
    mon = MonitorSeguridad()
    q = ik(pose_entrega(BOCA_NOMINAL, STANDOFF_ENTREGA), 0.0)
    tip, boca = punta(q), BOCA_NOMINAL.copy()
    t, t_det = 0.0, None
    while t < 3.0:
        t += T_CTRL
        d = dict(t=t, tip=tip, q=q.copy(), v_tip=40.0, t_ultimo_rostro=t, t_ultimo_latido=t, fase_aprox=True,
                 boca_est=boca, corriente=None, desviacion=None, paro_emergencia=False)
        if t >= T_FALLA:
            aplicar(d)
        estado, motivo = mon.evaluar(Entradas(**d))
        if estado is Estado.PARADA and t_det is None:
            t_det = t
    parada = t_det is not None
    reaccion = (t_det - T_FALLA) if parada else None
    q_otro = q + 0.1
    retiene = bool(parada and np.allclose(mon.comando(q_otro), mon.q_hold))
    return dict(escenario=nombre, umbral_s=umbral, reaccion_s=reaccion,
                retraso_sobre_umbral_s=(reaccion - umbral) if parada else None,
                parada=parada, retiene_posicion=retiene, motivo=mon.motivo,
                cumple=bool(parada and (reaccion - umbral) <= T_CTRL + 1e-9 and retiene))


def main():
    def perdida_rostro(d): d["t_ultimo_rostro"] = T_FALLA - T_CTRL
    def falla_com(d): d["t_ultimo_latido"] = T_FALLA - T_CTRL
    def distancia(d): d["boca_est"] = d["tip"] + np.array([25.0, 0, 0])
    def velocidad(d): d["v_tip"] = 80.0
    def limite(d): d["q"] = d["q"].copy(); d["q"][1] = Q_MAX[1] + 5 * D2R
    def corriente(d): d["corriente"] = np.array([0, 0, 1.5 * I_MAX, 0, 0])
    def desviacion(d): d["desviacion"] = 2 * DESV_MAX
    def emergencia(d): d["paro_emergencia"] = True
    casos = [("Pérdida de detección del rostro", perdida_rostro, TIMEOUT_ROSTRO),
             ("Falla de comunicación", falla_com, TIMEOUT_COM),
             ("Distancia mínima a la boca (25 mm)", distancia, 0.0),
             ("Velocidad de aproximación (80 mm/s)", velocidad, 0.0),
             ("Límite articular excedido", limite, 0.0),
             ("Corriente fuera de rango", corriente, 0.0),
             ("Desviación de posición", desviacion, 0.0),
             ("Paro de emergencia", emergencia, 0.0)]
    res = [correr(*c) for c in casos]
    control = correr("Sin falla (control)", lambda d: None, 0.0)
    control["cumple"] = not control["parada"]
    res.append(control)
    # rearme: la parada por pérdida de rostro solo se libera cuando el rostro vuelve a detectarse
    mon = MonitorSeguridad()
    q = ik(pose_entrega(BOCA_NOMINAL, STANDOFF_ENTREGA), 0.0); tip = punta(q)
    malo = Entradas(t=5.0, tip=tip, q=q, v_tip=0.0, t_ultimo_rostro=1.0, t_ultimo_latido=5.0, fase_aprox=True)
    mon.evaluar(malo)
    bueno = Entradas(t=5.1, tip=tip, q=q, v_tip=0.0, t_ultimo_rostro=5.1, t_ultimo_latido=5.1, fase_aprox=True)
    rearme = dict(rechazado_con_riesgo=(not mon.rearmar(malo)), aceptado_sin_riesgo=mon.rearmar(bueno))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pruebas_seguridad.json"), "w", encoding="utf8") as f:
        json.dump(dict(periodo_control_s=T_CTRL, t_parada_exigido_s=T_PARADA_MAX, casos=res, rearme=rearme), f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "pruebas_seguridad.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    for r in res:
        print(f"{'OK ' if r['cumple'] else 'FALLA'} {r['escenario']:<42} reacción={r['reaccion_s']}  umbral={r['umbral_s']}")
    print("rearme:", rearme)
    return res, rearme


if __name__ == "__main__":
    main()
