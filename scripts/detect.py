"""
ASL Sign Language Translator - Real-Time Detection Engine (Corrected)
"""
import cv2
import numpy as np
import json
import time
import argparse
from collections import deque, Counter
from pathlib import Path

import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE          = 224
SMOOTHING_WINDOW  = 10    
CONFIDENCE_THRESH = 0.5  
HOLD_FRAMES       = 15    
HAND_PADDING      = 40    
WORD_SUGGESTIONS = 3

# ─────────────────────────────────────────────
# NLP Suggestion Engine
# ─────────────────────────────────────────────
COMMON_WORDS = [
    "APPLE", "BALL", "CAT", "DOG", "EGG", "FISH", "GOOD", "HELLO", "ICE",
    "JAM", "KITE", "LOVE", "MORE", "NICE", "OPEN", "PEN", "QUEEN", "RED",
    "SUN", "TOP", "UP", "VERY", "WATER", "XRAY", "YELLOW", "ZERO",
    "ABLE", "ABOVE", "ACROSS", "AFTER", "AGAIN", "AGE", "BACK", "BAD",
    "BELOW", "BEST", "BIRD", "BLACK", "BLUE", "BOOK", "BOTH", "BOY",
    "BRING", "CALL", "CAME", "CITY", "COME", "COULD", "DAY", "DEAR",
    "DOWN", "DRAW", "EACH", "EARLY", "EVEN", "EVER", "EYES", "FACE",
    "FAST", "FEEL", "FEET", "FEW", "FIND", "FIRE", "FIRST", "FIVE",
    "FOOD", "FOR", "FROM", "FULL", "GAME", "GAVE", "GIVE", "GLAD",
    "GOING", "GONE", "GOT", "GREAT", "GREEN", "GROW", "HAND", "HARD",
    "HAS", "HAVE", "HEAD", "HIGH", "HILL", "HOME", "HOW", "HURT",
]

def get_word_suggestions(prefix: str, n: int = WORD_SUGGESTIONS) -> list[str]:
    prefix = prefix.upper()
    if not prefix:
        return []
    return [w for w in COMMON_WORDS if w.startswith(prefix)][:n]

# ─────────────────────────────────────────────
# Model + Label Map Loader
# ─────────────────────────────────────────────
def load_model_and_labels(model_path: str):
    model = tf.keras.models.load_model(model_path)
    label_path = Path(model_path).parent / "label_map.json"
    with open(label_path) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    return model, idx_to_label

# ─────────────────────────────────────────────
# Hand Detector (MediaPipe)
# ─────────────────────────────────────────────
class HandDetector:
    def __init__(self):
        self.mp_hands    = mp.solutions.hands
        self.mp_draw     = mp.solutions.drawing_utils
        self.hands       = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )

    def detect(self, frame_rgb: np.ndarray):
        return self.hands.process(frame_rgb)

    def get_hand_bbox(self, result, frame_shape, padding=HAND_PADDING):
        if not result.multi_hand_landmarks:
            return None
        h, w = frame_shape[:2]
        lms = result.multi_hand_landmarks[0].landmark
        xs  = [int(lm.x * w) for lm in lms]
        ys  = [int(lm.y * h) for lm in lms]
        x1 = max(0,     min(xs) - padding)
        y1 = max(0,     min(ys) - padding)
        x2 = min(w - 1, max(xs) + padding)
        y2 = min(h - 1, max(ys) + padding)
        return (x1, y1, x2, y2)

    def draw_landmarks(self, frame_bgr: np.ndarray, result):
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
                )

# ─────────────────────────────────────────────
# Prediction Smoother
# ─────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.window   = window
        self.history  = deque(maxlen=window)

    def update(self, label: str, confidence: float) -> tuple[str, float]:
        if confidence >= CONFIDENCE_THRESH:
            self.history.append(label)
        if not self.history:
            return ("", 0.0)
        counts       = Counter(self.history)
        best_label   = counts.most_common(1)[0][0]
        smoothed_conf = counts[best_label] / len(self.history)
        return best_label, smoothed_conf

    def reset(self):
        self.history.clear()

# ─────────────────────────────────────────────
# Frame Preprocessor
# ─────────────────────────────────────────────
def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)

# ─────────────────────────────────────────────
# HUD Renderer
# ─────────────────────────────────────────────
def draw_hud(frame, letter, conf, word, suggestions, fps, hand_visible):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    status_color = (0, 230, 100) if hand_visible else (0, 60, 200)
    status_text  = "HAND DETECTED" if hand_visible else "NO HAND"
    cv2.circle(frame, (20, 20), 8, status_color, -1)
    cv2.putText(frame, status_text, (35, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
    letter_display = letter if letter else "—"
    cv2.putText(frame, letter_display, (30, h - 80),
                cv2.FONT_HERSHEY_DUPLEX, 3.8, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, letter_display, (30, h - 80),
                cv2.FONT_HERSHEY_DUPLEX, 3.8, (0, 230, 200), 2, cv2.LINE_AA)
    if letter:
        bar_x, bar_y, bar_w, bar_h = 140, h - 120, 200, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * conf)
        bar_color = (0, 210, 80) if conf > 0.85 else (0, 160, 230)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"{conf*100:.0f}%", (bar_x + bar_w + 8, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "Confidence", (bar_x, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    word_display = word if word else "_"
    cv2.putText(frame, "WORD:", (350, h - 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, word_display, (350, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.40, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, word_display, (350, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.40, (255, 220, 80), 1, cv2.LINE_AA)
    if suggestions:
        cv2.putText(frame, "Suggestions:", (350, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
        sx = 460
        for sug in suggestions:
            cv2.putText(frame, sug, (sx, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1, cv2.LINE_AA)
            sx += len(sug) * 13 + 18
    controls = "[SPACE] add space  [BACKSPACE] delete  [ENTER] clear word  [Q] quit"
    cv2.putText(frame, controls, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────
# Main Detect Loop (Corrected Hold Logic)
# ─────────────────────────────────────────────
def run_detection(model_path: str, camera_index: int = 0):
    print("🔄 Loading model …")
    model, idx_to_label = load_model_and_labels(model_path)
    print(f"✅ Model loaded. Classes: {list(idx_to_label.values())}")

    detector = HandDetector()
    smoother = PredictionSmoother(window=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    word          = ""
    sentence      = ""
    current_letter = ""
    stable_count  = 0
    last_appended = ""

    fps_counter = deque(maxlen=30)

    print("\n📷 Starting real-time ASL detection …")
    print("   Controls: SPACE=space | BACKSPACE=delete | ENTER=clear | Q=quit\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result      = detector.detect(frame_rgb)
        hand_visible = result.multi_hand_landmarks is not None
        bbox         = detector.get_hand_bbox(result, frame.shape)

        letter, conf = "", 0.0

        if bbox:
            x1, y1, x2, y2 = bbox
            roi = frame_rgb[y1:y2, x1:x2]

            if roi.size > 0:
                inp          = preprocess_roi(roi)
                preds        = model.predict(inp, verbose=0)[0]
                top_idx      = np.argmax(preds)
                raw_conf     = float(preds[top_idx])
                raw_label    = idx_to_label[top_idx]
                letter, conf = smoother.update(raw_label, raw_conf)

                # رسم المربع
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 200), 2)
                cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), (0, 230, 200), -1)
                cv2.putText(frame, f"{letter} {conf*100:.0f}%", (x1 + 4, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

            # ✅ المنطق الصحيح للثبات (HOLD_FRAMES)
            if letter == current_letter and letter != "":
                stable_count += 1
            else:
                stable_count   = 0
                current_letter = letter

            if stable_count >= HOLD_FRAMES and letter != last_appended:
                word          += letter
                last_appended  = letter
                stable_count   = 0  # إعادة تعيين العداد بعد الإضافة
        else:
            smoother.reset()
            last_appended = ""

        detector.draw_landmarks(frame, result)
        suggestions = get_word_suggestions(word)

        fps_counter.append(time.time() - t0)
        fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

        frame = draw_hud(frame, letter, conf, word, suggestions, fps, hand_visible)
        cv2.imshow("ASL Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:
            if word:
                sentence += word + " "
                print(f"📝 Word added: {word}  |  Sentence: {sentence.strip()}")
                word = ""
                last_appended = ""
        elif key == 8:
            word = word[:-1]
            last_appended = word[-1] if word else ""
        elif key == 13:
            word = ""
            last_appended = ""

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📝 Final sentence: {sentence.strip()}")

# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Real-Time Detector")
    parser.add_argument("--model",  default="model/saved/asl_model_final.keras")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_detection(args.model, args.camera)