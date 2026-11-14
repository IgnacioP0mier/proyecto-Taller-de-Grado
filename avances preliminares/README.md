# Avances preliminares: brazo robótico asistencial de 5 GDL

Ignacio Eduardo Pomier Sevilla. Todo es simulación: las medidas de `config.py` son valores de partida.

## 1. Modelado y simulaciones cinemáticas

| Entregable | Archivos |
|---|---|
| Modelo Denavit-Hartenberg (5 GDL y tabla D-H) | `reporte/figura_5gdl_dh.png` |
| Simulación del espacio de trabajo alcanzable | `espacio_trabajo.py` → `resultados/fig_espacio_trabajo.png`, `espacio_trabajo.json`, `nube_espacio_trabajo.csv` |

## 2. Código fuente

| Entregable | Archivos |
|---|---|
| Procesamiento de imágenes con OpenCV (detección y estimación 3D de la boca) | `vision/vision_boca.py` (con sensor ToF), `vision/boca_webcam.py` (solo cámara), `vision/geom_boca.py`, `vision/calibrar_camara.py`, `vision/tof_esp32.ino` |
| Cinemática directa e inversa | `cinematica.py` |
| Control de trayectoria | `trayectoria.py`, `simulacion_ciclo.py` |
| Lógica del sistema de seguridad | `seguridad.py`, `pruebas_seguridad.py` |
| Parámetros (medidas, límites, escenario) | `config.py` |

## Cómo correrlo

```bash
pip install -r requirements.txt
python3 tests_cinematica.py     # verificación de la cinemática
python3 espacio_trabajo.py      # espacio de trabajo
python3 simulacion_ciclo.py     # ciclo con control de trayectoria
python3 pruebas_seguridad.py    # 8 fallas inyectadas
python3 vision/boca_webcam.py   # detección de la boca con la webcam ('c' fija la referencia, 'q' sale)
```

Los resultados se guardan en `resultados/`.
