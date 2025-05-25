import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class IndexFingerSwipeTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.last_pos = None  # (x, y)
        self.swipe_threshold_x = 0.05  # horizontal swipe sensitivity
        self.swipe_threshold_y = 0.05  # vertical swipe sensitivity
        self.cooldown = 0.5  # seconds between triggers
        self.last_trigger = time.time()

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None
        hand_landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            current_pos = (index_tip.x, index_tip.y)

            if self.last_pos:
                dx = current_pos[0] - self.last_pos[0]
                dy = current_pos[1] - self.last_pos[1]

                # Invert horizontal swipe direction here
                if abs(dx) > self.swipe_threshold_x and abs(dx) > abs(dy):
                    gesture = "left" if dx > 0 else "right"  # inverted

                elif abs(dy) > self.swipe_threshold_y and abs(dy) > abs(dx):
                    gesture = "down" if dy > 0 else "up"

            self.last_pos = current_pos

        else:
            self.last_pos = None  # reset if no hand detected

        return gesture, hand_landmarks

def trigger_key(gesture, tracker):
    now = time.time()
    if gesture and (now - tracker.last_trigger > tracker.cooldown):
        if gesture == "left":
            pyautogui.press("left")
        elif gesture == "right":
            pyautogui.press("right")
        elif gesture == "up":
            pyautogui.press("up")
        elif gesture == "down":
            pyautogui.press("down")
        tracker.last_trigger = now

def main():
    cap = cv2.VideoCapture(0)
    tracker = IndexFingerSwipeTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, landmarks = tracker.detect_gesture(frame)

        if landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        if gesture:
            cv2.putText(frame, gesture.upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            trigger_key(gesture, tracker)

        cv2.imshow("Index Finger Swipe Controls (Opposite Left/Right)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
