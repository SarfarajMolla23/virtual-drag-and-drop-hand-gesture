import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)  # Start with camera index 0
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)
colorR = (255, 0, 255)

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.dragging = False

    def update(self, cursor, cursor_distance):
        cx, cy = self.posCenter
        w, h = self.size

        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            if cursor_distance < 30:
                self.dragging = True

        if self.dragging:
            self.posCenter = cursor

        if cursor_distance > 40:
            self.dragging = False


rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Retrying...")
        continue

    img = cv2.flip(img, 1)
    hands = detector.findHands(img, draw=True)  # Detect hands and draw landmarks

    if hands and len(hands) > 0:
        # Debugging: Print the structure of the first hand
        print("Hand Data:", hands[0])

        # Extract the landmarks (assuming `hands[0]` is a list of landmarks)
        lmList = hands[0]['lmList'] if isinstance(hands[0], dict) and 'lmList' in hands[0] else hands[0]

        if len(lmList) > 8:  # Ensure there are enough landmarks
            l, _, _ = detector.findDistance(8, 12, img, draw=False)  # Index
            # tip and thumb tip distance
            cursor = lmList[8]  # Index finger tip position
            for rect in rectList:
                rect.update(cursor, l)

    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
