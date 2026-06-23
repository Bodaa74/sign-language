import cv2
import numpy as np
import json
import time
from pathlib import Path
from collections import deque, Counter
import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "model/saved/asl_model_final.keras"
LABEL_PATH = "model/saved/label_map.json"
CONFIDENCE_THRESHOLD = 0.5
HOLD_FRAMES = 20
SMOOTHING_WINDOW = 10

class ASLModelTester:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.idx_to_label = None
        self.label_map = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """تحميل الموديل وملف العلامات"""
        print("=" * 60)
        print("📦 LOADING ASL MODEL")
        print("=" * 60)
        
        try:
            print(f"\n🔍 Looking for model: {model_path}")
            model_file = Path(model_path)
            
            if not model_file.exists():
                print(f"❌ Model not found at: {model_path}")
                print("   Check if the path is correct")
                return False
            
            # Load model
            print("⏳ Loading TensorFlow model...")
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully!")
            
            # Model info
            print(f"\n📊 Model Details:")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            print(f"   Total layers: {len(self.model.layers)}")
            
            # Load label map
            label_path = model_file.parent / "label_map.json"
            print(f"\n🔍 Looking for labels: {label_path}")
            
            if not label_path.exists():
                print(f"⚠️  Label map not found!")
                print("   Creating default A-Z label map...")
                self.idx_to_label = {i: chr(65+i) for i in range(26)}
            else:
                with open(label_path) as f:
                    self.label_map = json.load(f)
                self.idx_to_label = {v: k for k, v in self.label_map.items()}
            
            print(f"✅ Labels loaded: {len(self.idx_to_label)} classes")
            print(f"   Classes: {list(self.idx_to_label.values())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """تجهيز الصورة للتنبؤ"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image: {image}")
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_single(self, image):
        """تنبؤ واحد من صورة"""
        if self.model is None:
            return None, 0.0
        
        try:
            input_tensor = self.preprocess_image(image)
            
            start_time = time.time()
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            prediction_time = (time.time() - start_time) * 1000
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            results = []
            for idx in top_indices:
                letter = self.idx_to_label.get(idx, f"Class_{idx}")
                confidence = float(predictions[idx])
                results.append((letter, confidence))
            
            return results, prediction_time
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def test_realtime_with_word(self):
        """اختبار فوري بالكاميرا مع كتابة الكلمة"""
        print("\n" + "=" * 60)
        print("📷 REAL-TIME ASL DETECTION WITH WORD FORMATION")
        print("=" * 60)
        
        if self.model is None:
            print("❌ Model not loaded!")
            return
        
        # Initialize MediaPipe
        print("\n🔧 Setting up MediaPipe...")
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Open camera
        print("📷 Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            hands.close()
            return
        
        print("✅ Camera ready!")
        print("\n" + "=" * 60)
        print("🎯 CONTROLS:")
        print("   ✋ Show hand sign → Automatic letter detection")
        print("   ⌨️  SPACE → Add space (complete word)")
        print("   ⌨️  BACKSPACE → Delete last letter")
        print("   ⌨️  ENTER → Clear current word")
        print("   ⌨️  C → Clear entire sentence")
        print("   ⌨️  Q → Quit")
        print("=" * 60)
        print("\n📝 Start signing! Form words with your hand signs.\n")
        
        # Word and sentence tracking
        current_word = ""
        sentence = ""
        current_letter = ""
        last_appended_letter = ""
        stable_count = 0
        
        # Smoothing
        prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        
        # FPS tracking
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect hands
                results = hands.process(frame_rgb)
                hand_detected = results.multi_hand_landmarks is not None
                
                detected_letter = ""
                confidence = 0.0
                top_predictions = []
                
                if hand_detected:
                    # Get hand bounding box
                    h, w = frame.shape[:2]
                    landmarks = results.multi_hand_landmarks[0].landmark
                    
                    x_coords = [int(lm.x * w) for lm in landmarks]
                    y_coords = [int(lm.y * h) for lm in landmarks]
                    
                    padding = 40
                    x1 = max(0, min(x_coords) - padding)
                    y1 = max(0, min(y_coords) - padding)
                    x2 = min(w, max(x_coords) + padding)
                    y2 = min(h, max(y_coords) + padding)
                    
                    # Extract ROI and predict
                    roi = frame_rgb[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        results_pred, pred_time = self.predict_single(roi)
                        
                        if results_pred:
                            raw_letter, raw_confidence = results_pred[0]
                            top_predictions = results_pred
                            
                            # Apply smoothing
                            if raw_confidence > CONFIDENCE_THRESHOLD:
                                prediction_history.append(raw_letter)
                            
                            if prediction_history:
                                letter_counts = Counter(prediction_history)
                                detected_letter = letter_counts.most_common(1)[0][0]
                                confidence = letter_counts[detected_letter] / len(prediction_history)
                            else:
                                detected_letter = raw_letter
                                confidence = raw_confidence
                            
                            # Smart letter stability for word building
                            if detected_letter == current_letter and detected_letter != "":
                                stable_count += 1
                            else:
                                stable_count = 0
                                current_letter = detected_letter
                            
                            # Add letter to word after holding
                            if stable_count >= HOLD_FRAMES and detected_letter != last_appended_letter:
                                current_word += detected_letter
                                last_appended_letter = detected_letter
                                stable_count = 0
                                print(f"✅ Letter '{detected_letter}' added → Word: {current_word}")
                            
                            # Draw hand landmarks
                            mp_drawing.draw_landmarks(
                                frame,
                                results.multi_hand_landmarks[0],
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2)
                            )
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                            
                            # Prediction label on box
                            cv2.rectangle(frame, (x1, y1-50), (x2, y1), (0, 255, 200), -1)
                            cv2.putText(frame, 
                                       f"{detected_letter} {confidence*100:.0f}%",
                                       (x1+5, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       0.7, (0, 0, 0), 2)
                            
                            # Stability progress bar
                            bar_width = x2 - x1
                            bar_height = 6
                            bar_y = y2 + 15
                            
                            cv2.rectangle(frame,
                                        (x1, bar_y),
                                        (x2, bar_y + bar_height),
                                        (40, 40, 40), -1)
                            
                            if stable_count > 0:
                                progress = min(stable_count / HOLD_FRAMES, 1.0)
                                fill_width = int(bar_width * progress)
                                
                                fill_color = (0, 255, 0) if progress >= 0.8 else (0, 200, 255)
                                cv2.rectangle(frame,
                                            (x1, bar_y),
                                            (x1 + fill_width, bar_y + bar_height),
                                            fill_color, -1)
                
                else:
                    # Reset when hand disappears
                    prediction_history.clear()
                    current_letter = ""
                    stable_count = 0
                
                # ── Draw HUD Interface ──
                self._draw_hud(frame, detected_letter, confidence, 
                              current_word, sentence, top_predictions,
                              hand_detected, stable_count)
                
                # ── FPS ──
                fps_counter.append(time.time() - t0)
                fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                
                # Show frame
                cv2.imshow('ASL Sign Language - Write Words', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == 32:  # SPACE - Complete word
                    if current_word:
                        sentence += current_word + " "
                        print(f"\n📝 Word completed: '{current_word}'")
                        print(f"📜 Sentence: {sentence.strip()}")
                        current_word = ""
                        last_appended_letter = ""
                        current_letter = ""
                        stable_count = 0
                elif key == 8:  # BACKSPACE - Delete last letter
                    if current_word:
                        current_word = current_word[:-1]
                        last_appended_letter = current_word[-1] if current_word else ""
                        print(f"⌫ Deleted → Word: {current_word}")
                elif key == 13:  # ENTER - Clear current word
                    current_word = ""
                    last_appended_letter = ""
                    current_letter = ""
                    stable_count = 0
                    print("🔄 Current word cleared")
                elif key == ord('c') or key == ord('C'):
                    current_word = ""
                    sentence = ""
                    last_appended_letter = ""
                    current_letter = ""
                    stable_count = 0
                    print("🗑️  Entire sentence cleared")
                
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final result
            print("\n" + "=" * 60)
            print("📜 FINAL RESULT")
            print("=" * 60)
            if sentence or current_word:
                final_text = sentence + current_word
                print(f"   {final_text.strip()}")
            else:
                print("   (No words formed)")
            print("=" * 60)
            print("✅ Test completed")
    
    def _draw_hud(self, frame, letter, confidence, current_word, sentence, 
                  top_predictions, hand_detected, stable_count):
        """رسم واجهة المستخدم مع الكلمة والجملة"""
        h, w = frame.shape[:2]
        
        # ── Semi-transparent bottom panel ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 200), (w, h), (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # ── Hand status indicator ──
        status_color = (0, 255, 100) if hand_detected else (0, 100, 255)
        status_text = "HAND DETECTED" if hand_detected else "NO HAND"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # ── Current detected letter (large display) ──
        if hand_detected and letter:
            letter_display = letter
            cv2.putText(frame, letter_display, (30, h - 130),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 6)
            cv2.putText(frame, letter_display, (30, h - 130),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 200), 3)
            
            # Confidence bar
            bar_x, bar_y = 200, h - 160
            bar_w, bar_h = 250, 18
            
            cv2.rectangle(frame, (bar_x, bar_y), 
                        (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            
            fill = int(bar_w * confidence)
            bar_color = (0, 220, 80) if confidence > 0.8 else (0, 170, 240)
            cv2.rectangle(frame, (bar_x, bar_y), 
                        (bar_x + fill, bar_y + bar_h), bar_color, -1)
            
            cv2.putText(frame, f"{confidence*100:.0f}%", 
                       (bar_x + bar_w + 10, bar_y + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            
            # Stability indicator
            if stable_count > 0:
                stability_pct = min(stable_count / HOLD_FRAMES * 100, 100)
                cv2.putText(frame, f"Stable: {stability_pct:.0f}%", 
                           (bar_x, bar_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        else:
            cv2.putText(frame, "—", (30, h - 130),
                       cv2.FONT_HERSHEY_DUPLEX, 5, (100, 100, 100), 4)
        
        # ── Current word being built ──
        cv2.putText(frame, "CURRENT WORD:", (30, h - 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        
        word_display = current_word if current_word else "_"
        cv2.putText(frame, word_display, (30, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(frame, word_display, (30, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 200, 50), 1)
        
        # ── Full sentence ──
        if sentence:
            cv2.putText(frame, "SENTENCE:", (30, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
            
            sentence_display = sentence + current_word
            cv2.putText(frame, sentence_display, (120, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # ── Top 3 predictions (top-right corner) ──
        if top_predictions and len(top_predictions) > 1:
            pred_x = w - 200
            cv2.putText(frame, "Top Predictions:", (pred_x, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            for i, (pred_letter, pred_conf) in enumerate(top_predictions[:3]):
                color = (0, 255, 200) if i == 0 else (150, 150, 150)
                cv2.putText(frame, f"{pred_letter}: {pred_conf*100:.0f}%", 
                           (pred_x, 85 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # ── Controls reminder ──
        controls = "[SPACE] Complete Word | [BACKSPACE] Delete | [ENTER] Clear Word | [C] Clear All | [Q] Quit"
        cv2.putText(frame, controls, (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

# ─────────────────────────────────────────────
# Quick Model Check
# ─────────────────────────────────────────────

def quick_model_check():
    """فحص سريع للموديل"""
    print("🔍 Quick Model Check\n")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
        
        random_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        
        start = time.time()
        prediction = model.predict(random_input, verbose=0)
        pred_time = (time.time() - start) * 1000
        
        print(f"✅ Prediction successful!")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Prediction time: {pred_time:.1f}ms")
        print(f"   Number of classes: {prediction.shape[1]}")
        
        top_idx = np.argmax(prediction[0])
        top_conf = float(prediction[0][top_idx])
        print(f"   Top class: {top_idx} ({top_conf*100:.1f}%)")
        
        print("\n🎉 Model is ready to use!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    tester = ASLModelTester()
    if tester.model:
        tester.test_realtime_with_word()
            