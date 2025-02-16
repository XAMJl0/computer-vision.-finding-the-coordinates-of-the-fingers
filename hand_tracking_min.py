import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

pTime = 0
cTime = 0

def count_fingers(hand_landmarks, is_right_hand):
    fingers = []

    # Проверка ориентации руки (ладонь вверх или вниз)
    palm_direction = is_hand_facing_up(hand_landmarks, is_right_hand)

    for id in range(1, 5):
        if palm_direction:  # Ладонь вверх
            if hand_landmarks.landmark[tipIds[id]].y < hand_landmarks.landmark[tipIds[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Ладонь вниз
            if hand_landmarks.landmark[tipIds[id]].y > hand_landmarks.landmark[tipIds[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

    # Большой палец
    if is_right_hand:
        if palm_direction:
            if hand_landmarks.landmark[tipIds[0]].x > hand_landmarks.landmark[tipIds[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
    else:
        if palm_direction:
            if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tipIds[0]].x > hand_landmarks.landmark[tipIds[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

    return fingers.count(1)

def is_hand_facing_up(hand_landmarks, is_right_hand):
    wrist = hand_landmarks.landmark[0]
    middle_finger_tip = hand_landmarks.landmark[12]

    if is_right_hand:
        return middle_finger_tip.y < wrist.y
    else:
        return middle_finger_tip.y < wrist.y

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    totalFingers = 0

    if results.multi_hand_landmarks:
        for handLms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_info.classification[0].label
            is_right_hand = hand_label == 'Right'
            
            num_fingers = count_fingers(handLms, is_right_hand)
            totalFingers += num_fingers

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Вывод информации о количестве отображаемых пальцев и FPS на экран
    cv2.putText(img, "FPS = " + str(int(fps)) + " | Fingers: " + str(totalFingers), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)