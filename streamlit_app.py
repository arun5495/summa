import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def pixel_to_cm(pixel_dist, ref_width=10.0, focal=850):
    return (ref_width * focal) / pixel_dist

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    
    h, w, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Thumb tip (4) vs Index tip (8)
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            
            t_point = (int(thumb.x * w), int(thumb.y * h))
            i_point = (int(index.x * w), int(index.y * h))
            
            pixel_dist = np.sqrt((thumb.x-index.x)**2 + (thumb.y-index.y)**2) * w
            cm_dist = pixel_to_cm(pixel_dist)
            
            # Draw AR
            cv2.line(frame, t_point, i_point, (0,255,0), 3)
            cv2.circle(frame, t_point, 8, (0,0,255), -1)
            cv2.circle(frame, i_point, 8, (0,0,255), -1)
            
            cv2.putText(frame, f"{cm_dist:.1f}cm", (t_point[0], t_point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    cv2.imshow("AR Ruler", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
