"""WebSocket prediction smoother — majority vote over a sliding window."""

from collections import deque, Counter

SMOOTHING_WINDOW  = 8
CONFIDENCE_THRESH = 0.70


class WebSocketSmoother:
    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, label: str, confidence: float) -> tuple[str, float]:
        if confidence >= CONFIDENCE_THRESH and label:
            self.history.append(label)
        if not self.history:
            return "", 0.0
        counts = Counter(self.history)
        best   = counts.most_common(1)[0][0]
        return best, counts[best] / len(self.history)

    def reset(self):
        self.history.clear()
