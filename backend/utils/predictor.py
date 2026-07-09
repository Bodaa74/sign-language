"""
ASL Predictor — Model inference utility shared by HTTP and WS routes.
"""

import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path

IMG_SIZE          = 224
CONFIDENCE_THRESH = 0.50
HAND_PADDING      = 45


class ASLPredictor:
    """Wraps model loading, hand detection, and single-image inference."""

    def __init__(self, model_path: str, label_path: str):
        self.model     = tf.keras.models.load_model(model_path)
        with open(label_path) as f:
            label_map  = json.load(f)
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.labels        = list(label_map.keys())
        self.num_classes   = len(self.labels)

        # MediaPipe hand detector
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.50,
        )

    def _extract_hand_roi(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Detect hand via MediaPipe and return cropped RGB ROI, or None."""
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result  = self._hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            return None

        lms = result.multi_hand_landmarks[0].landmark
        xs  = [int(lm.x * w) for lm in lms]
        ys  = [int(lm.y * h) for lm in lms]
        x1  = max(0,     min(xs) - HAND_PADDING)
        y1  = max(0,     min(ys) - HAND_PADDING)
        x2  = min(w - 1, max(xs) + HAND_PADDING)
        y2  = min(h - 1, max(ys) + HAND_PADDING)

        roi = img_rgb[y1:y2, x1:x2]
        return roi if roi.size > 0 else None

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)

    def predict(self, img_bgr: np.ndarray) -> dict:
        """
        Run inference on an image.
        Falls back to full-image crop if no hand detected.
        """
        roi = self._extract_hand_roi(img_bgr)
        hand_detected = roi is not None

        if roi is None:
            # Fallback: use center crop of full image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w    = img_rgb.shape[:2]
            s       = min(h, w)
            y0, x0  = (h - s) // 2, (w - s) // 2
            roi     = img_rgb[y0:y0+s, x0:x0+s]

        inp   = self._preprocess(roi)
        preds = self.model.predict(inp, verbose=0)[0]

        top3_idx = np.argsort(preds)[::-1][:3]
        top3     = [
            {"letter": self.idx_to_label[int(i)], "confidence": round(float(preds[i]), 4)}
            for i in top3_idx
        ]

        best_idx  = int(top3_idx[0])
        best_conf = float(preds[best_idx])

        return {
            "letter":     self.idx_to_label[best_idx] if hand_detected and best_conf >= CONFIDENCE_THRESH else "",
            "confidence": round(best_conf, 4),
            "top3":       top3,
            "candidate_letter": self.idx_to_label[best_idx],
            "candidate_confidence": round(best_conf, 4),
            "hand_detected": hand_detected,
        }
