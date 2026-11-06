import cv2
import numpy as np

VIDEO = "video autopilot.mp4"

Y_SUPERIOR = 0.45
Y_INFERIOR = 0.75
X_IZQUIERDA = 0.40
ANGULO_MIN = 15
ANGULO_MAX = 75
MARGEN = 0.08
MAX_PERDIDOS = 15

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("No se puede abrir el vídeo")

marca = None
perdidos = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alto, ancho = frame.shape[:2]
    y_sup = int(alto * Y_SUPERIOR)
    y_inf = int(alto * Y_INFERIOR)
    x_izq = int(ancho * X_IZQUIERDA)

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, 100, 200)
    roi = bordes[y_sup:y_inf, x_izq:]

    lineas = cv2.HoughLinesP(roi, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=25)

    candidatas = []
    if lineas is not None:
        for x1, y1, x2, y2 in lineas.reshape(-1, 4):
            x1 += x_izq
            x2 += x_izq
            y1 += y_sup
            y2 += y_sup
            if x2 == x1 or y2 == y1:
                continue
            angulo = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
            if angulo < ANGULO_MIN or angulo > ANGULO_MAX:
                continue
            pendiente = (x2 - x1) / (y2 - y1)
            x_sup = x1 + pendiente * (y_sup - y1)
            x_inf = x1 + pendiente * (y_inf - y1)
            longitud = np.hypot(x2 - x1, y2 - y1)
            candidatas.append((x_sup, x_inf, longitud))

    if marca is not None:
        candidatas = [c for c in candidatas
                      if abs(c[0] - marca[0]) < MARGEN * ancho
                      and abs(c[1] - marca[1]) < MARGEN * ancho]

    detectada = len(candidatas) > 0
    if detectada:
        if marca is None:
            mejor = max(candidatas, key=lambda c: c[2])
            marca = np.array([mejor[0], mejor[1]])
        else:
            pesos = [c[2] for c in candidatas]
            medida = np.array([
                np.average([c[0] for c in candidatas], weights=pesos),
                np.average([c[1] for c in candidatas], weights=pesos),
            ])
            marca = (marca + medida) / 2
        perdidos = 0
    else:
        perdidos += 1
        if perdidos > MAX_PERDIDOS:
            marca = None

    if marca is not None:
        color = (0, 255, 0) if detectada else (0, 255, 255)
        cv2.line(frame, (int(marca[0]), y_sup), (int(marca[1]), y_inf), color, 6)

    cv2.imshow("Seguimiento de la marca vial derecha", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
