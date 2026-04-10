#Remember to enter these 2 command before running the code:
#source ~/cv_env/bin/activate
#python detection_main.py

import cv2
from symbol_detector import SymbolDetector

detector = SymbolDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect(frame)

    for obj in results:
        x, y, w, h = obj["bbox"]

        label = obj["shape"]
        if obj["direction"]:
            label += " " + obj["direction"]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        print(label)

    cv2.imshow("Symbol Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()