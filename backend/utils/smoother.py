"""WebSocket prediction smoother — majority vote over a sliding window,
with fast-switch detection so a genuinely new sign isn't dragged down
by stale frames from the previous letter."""

from collections import deque, Counter

SMOOTHING_WINDOW  = 8
CONFIDENCE_THRESH = 0.50
SWITCH_THRESHOLD  = 3   # consecutive raw frames of a NEW label needed to force a switch


class WebSocketSmoother:
    def __init__(self, window: int = SMOOTHING_WINDOW, switch_threshold: int = SWITCH_THRESHOLD):
        self.window = window
        self.switch_threshold = switch_threshold
        self.history = deque(maxlen=window)
        self.current_label = ""
        self.consecutive_diff = 0

    def update(self, label: str, confidence: float) -> tuple[str, float]:
        if confidence < CONFIDENCE_THRESH or not label:
            self.reset()
            return "", 0.0

        # Detect a sustained change in the RAW prediction (not the smoothed one).
        if self.current_label and label != self.current_label:
            self.consecutive_diff += 1
            if self.consecutive_diff >= self.switch_threshold:
                # The user has clearly moved to a new sign — flush stale frames
                # instead of letting them drag the majority vote.
                self.history.clear()
                self.consecutive_diff = 0
        else:
            self.consecutive_diff = 0

        self.history.append(label)
        if not self.history:
            return "", 0.0

        counts = Counter(self.history)
        best, best_count = counts.most_common(1)[0]
        self.current_label = best
        return best, best_count / len(self.history)

    def reset(self):
        self.history.clear()
        self.current_label = ""
        self.consecutive_diff = 0