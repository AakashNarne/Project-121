import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmark.landmark]

            # Draw circles on finger tips
            for tip in finger_tips:
                x, y = lm_list[tip]
                cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)

            # Draw circle on thumb tip
            x_thumb, y_thumb = lm_list[thumb_tip]
            cv2.circle(img, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)

            # Draw landmarks and connections
            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
