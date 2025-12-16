import time
import threading
import queue

import cv2
import pyttsx3

from tracker_core.config import AppConfig
from .attention_inference import SystemState, StateOutput


class TTSWorker:
    def __init__(self, rate: int, volume: float):
        self.rate = rate
        self.volume = volume
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def speak(self, text: str):
        self.q.put(text)

    def close(self):
        self._stop.set()
        self.q.put("")
        self.thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            text = self.q.get()
            if self._stop.is_set():
                break
            if not text:
                continue
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception:
                pass


class FeedbackManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self._prev_state = SystemState.OK
        self._last_spoken_state = SystemState.OK
        self._last_spoken_time = 0.0

        self.tts = None
        if self.config.enable_tts:
            self.tts = TTSWorker(rate=self.config.tts_rate, volume=self.config.tts_volume)

    def close(self):
        if self.tts is not None:
            self.tts.close()

    @staticmethod
    def _draw_transparent_box(img, x1, y1, x2, y2, bgr, alpha=0.55):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    @staticmethod
    def _put_text_outline(img, text, org, font, scale, color, thickness=2, outline=4):
        x, y = org
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _centered_x(img_w: int, text: str, font, scale: float, thickness: int) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        return max(0, (img_w - tw) // 2)

    def _state_style(self, state: SystemState):
        if state == SystemState.DISTRACTED:
            return ("DISTRACTED", (30, 30, 30), (0, 255, 255))
        if state == SystemState.DROWSY:
            return ("DROWSY", (30, 30, 30), (0, 165, 255))
        if state == SystemState.POSTURE_RISK:
            return ("POSTURE", (30, 30, 30), (0, 0, 255))
        return ("OK", (30, 30, 30), (0, 255, 0))

    def _mode_style(self, mode: str):
        m = (mode or "").lower()
        if "high" in m:
            return (0, 140, 255), (0, 0, 0)
        if "office" in m:
            return (255, 200, 0), (0, 0, 0)
        return (0, 220, 0), (0, 0, 0)


    @staticmethod
    def _state_to_message(state: SystemState) -> str:
        if state == SystemState.DISTRACTED:
            return "Please bring your attention back to the screen."
        if state == SystemState.DROWSY:
            return "You seem drowsy. Consider taking a short break."
        if state == SystemState.POSTURE_RISK:
            return "Please correct your posture."
        return ""

    def draw_hud(self, frame, state_output: StateOutput):
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        self._draw_transparent_box(frame, 0, 0, w, 56, (0, 0, 0), alpha=0.35)

        label, pill_bg, pill_text = self._state_style(state_output.state)
        pill = f"STATUS: {label}"

        scale = 0.75
        thickness = 2
        (tw, th), _ = cv2.getTextSize(pill, font, scale, thickness)
        pad_x, pad_y = 14, 10
        x1, y1 = 12, 10
        x2, y2 = x1 + tw + 2 * pad_x, y1 + th + 2 * pad_y

        self._draw_transparent_box(frame, x1, y1, x2, y2, pill_bg, alpha=0.70)
        self._put_text_outline(frame, pill, (x1 + pad_x, y2 - pad_y), font, scale, pill_text, thickness=2, outline=4)

        reasons = ", ".join(state_output.reasons) if state_output.reasons else ""
        mode = getattr(self.config, "active_mode", "Study")
        badge_bg, badge_text = self._mode_style(mode)
        badge = f"MODE: {mode}"

        b_scale, b_thick = 0.70, 2
        (bw, bh), _ = cv2.getTextSize(badge, font, b_scale, b_thick)
        bpad_x, bpad_y = 14, 10

        bx2 = w - 12
        bx1 = bx2 - (bw + 2 * bpad_x)
        by1 = 10
        by2 = by1 + bh + 2 * bpad_y

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, thickness=-1)
        cv2.putText(frame, badge, (bx1 + bpad_x, by2 - bpad_y), font, b_scale, badge_text, b_thick, cv2.LINE_AA)

        msg = self._state_to_message(state_output.state) if state_output.state != SystemState.OK else ""
        if msg:
            toast_scale = 0.70
            toast_thick = 2
            (mw, mh), _ = cv2.getTextSize(msg, font, toast_scale, toast_thick)
            pad = 16

            box_w = min(w - 60, mw + 2 * pad)
            box_h = mh + 2 * pad

            tx1 = (w - box_w) // 2
            tx2 = tx1 + box_w
            ty1 = 62
            ty2 = ty1 + box_h

            self._draw_transparent_box(frame, tx1, ty1, tx2, ty2, (15, 15, 15), alpha=0.55)

            text_x = self._centered_x(box_w, msg, font, toast_scale, toast_thick) + tx1
            text_y = ty2 - pad
            self._put_text_outline(frame, msg, (text_x, text_y), font, toast_scale, (255, 255, 255), thickness=1, outline=4)

        return frame

    def maybe_alert(self, state_output: StateOutput):
        now = time.time()
        cur = state_output.state

        if cur == SystemState.OK:
            self._prev_state = cur
            self._last_spoken_state = SystemState.OK
            return None

        entered_alert = (self._prev_state == SystemState.OK and cur != SystemState.OK)
        changed_alert_type = (cur != self._last_spoken_state)
        cooled_down = (now - self._last_spoken_time) >= self.config.alert_cooldown_seconds

        if self.tts is not None and (entered_alert or (changed_alert_type and cooled_down)):
            msg = self._state_to_message(cur)
            if msg:
                self.tts.speak(msg)
                self._last_spoken_state = cur
                self._last_spoken_time = now
                self._prev_state = cur
                return {"event": "ALERT_FIRED", "state": cur.name, "message": msg}

        self._prev_state = cur
        return None