import cv2
import numpy as np

video_path = "autopilot/Pra╠üctica Autopilot.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=150, maxLineGap=10
    )

    if lines is not None:
        for line in lines:
            if len(line) == 4:
                x1, y1, x2, y2 = line
            else:
                x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            if dx != 0:
                slope = dy / dx
                midpoint_x = (x1 + x2) / 2

                if slope > 0.3 and midpoint_x > w * 0.45:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Seguimiento del carril derecho", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
