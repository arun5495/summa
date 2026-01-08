import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand dots
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    cv2.imshow("AR Ruler", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
