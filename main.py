import time
import cv2

from tracker_core.config import load_config, AppConfig
from tracker_core.gaze_tracker import GazeTracker
from tracker_core.mode_manager import ModeManager

from wellness.posture_module import PostureModule
from wellness.attention_inference import AttentionInference, SystemState
from wellness.feedback import FeedbackManager
from wellness.logging_utils import EventLogger
from wellness.calibration import CalibrationManager


def reinit_modules(config: AppConfig, gaze_tracker, posture_module, inference, feedback):
    posture_module.config = config
    inference.config = config
    feedback.config = config
    return posture_module, inference, feedback


def main():
    config = load_config()
    mode_manager = ModeManager()

    gaze_tracker = GazeTracker(config)
    posture_module = PostureModule(config)
    inference = AttentionInference(config)
    feedback = FeedbackManager(config)
    logger = EventLogger(config)
    calibration = CalibrationManager(config)

    prev_time = time.time()
    prev_state = SystemState.OK

    baseline = calibration.run(gaze_tracker, posture_module, seconds=3.0)
    logger.log_event("CALIBRATION_DONE", "initial_startup")

    try:
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            frame_data = gaze_tracker.read()
            if frame_data is None:
                break

            posture_data = posture_module.update(frame_data.frame_bgr)
            state_output = inference.update(frame_data, posture_data, dt)

            if state_output.state != prev_state:
                logger.log_event("STATE_CHANGE", f"{prev_state.name}->{state_output.state.name}")
                prev_state = state_output.state

            frame = feedback.draw_hud(frame_data.frame_bgr, state_output)

            alert_evt = feedback.maybe_alert(state_output)
            if alert_evt:
                logger.log_event("ALERT_FIRED", alert_evt["state"], alert_evt)

            logger.log_frame(config, frame_data, posture_data, state_output)

            cv2.imshow("WellnessWatch", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.log_event("KEYPRESS", "q_quit")
                break

            if key == ord("c"):
                logger.log_event("KEYPRESS", "c_calibrate")
                _ = calibration.run(gaze_tracker, posture_module, seconds=3.0)
                logger.log_event("CALIBRATION_DONE", "user_pressed_c")

            if key == ord("m"):
                logger.log_event("KEYPRESS", "m_mode_cycle")
                old = config.active_mode
                config = mode_manager.cycle_mode(config.active_mode, config.available_modes)
                logger.log_event("MODE_CHANGE", f"{old}->{config.active_mode}")
                posture_module, inference, feedback = reinit_modules(
                    config, gaze_tracker, posture_module, inference, feedback
                )

    finally:
        gaze_tracker.release()
        posture_module.close()
        feedback.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()