import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Инициализация модели для обнаружения рук
hands = mp_hands.Hands()

# Запуск видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация переменных для подсчета FPS
start_time = cv2.getTickCount()
fps = 0
num_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при получении кадра")
        continue

    # Конвертация кадра в формат RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаружение рук на кадре
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Получение координат средней точки руки для определения левой или правой руки
            cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            if cx < frame.shape[1] // 2:
                hand_side = "Left Hand"
                color = (255, 255, 255)  # Белый цвет для левой руки
                connections_color = (255, 255, 255)  # Белые линии для левой руки
            else:
                hand_side = "Right Hand"
                color = (0, 0, 0)  # Черный цвет для правой руки
                connections_color = (0, 0, 255)  # Красные линии для правой руки

            # Вывод информации о руке возле трекера пальцев
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=connections_color, thickness=2, circle_radius=4))
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0:
                    cv2.putText(frame, hand_side, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Визуализация ключевых точек и связей
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Подсчет FPS
    num_frames += 1
    current_time = cv2.getTickCount()
    fps = num_frames / ((current_time - start_time) / cv2.getTickFrequency())

    # Отображение FPS в углу окна
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра с информацией о руке
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()