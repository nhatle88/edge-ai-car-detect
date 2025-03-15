import threading
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time
import datetime
import subprocess
import math
import os

# Global dictionary and lock for frames
display_frames = {}
display_lock = threading.Lock()
stop_event = threading.Event()


class Notifier:
    def __init__(self, cooldown=5):
        self.cooldown = cooldown  # seconds between notifications
        self.last_spoken = 0

    def speak(self, message):
        current_time = time.time()
        if current_time - self.last_spoken >= self.cooldown:
            subprocess.Popen(['say', message])
            self.last_spoken = current_time


class PersonDetection:
    def __init__(self, confidence_threshold=0.4, roi=None, movement_threshold=5.0, history_size=3):
        # Use MPS if available (for Apple Silicon), otherwise CPU.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.person_class = 0  # Person class in YOLOv8
        self.notifier = Notifier()
        # roi is a tuple (x, y, w, h); if None, process full frame.
        self.roi = roi
        self.movement_threshold = movement_threshold
        self.previous_frame = None
        self.frame_history = []
        self.history_size = history_size
        self.consecutive_detections = {}  # Track consecutive detections by position

    def _inside_roi(self, box):
        if self.roi is None:
            return True
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        rx, ry, rw, rh = self.roi
        return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)

    def detect_persons(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, verbose=False)

        # Initialize previous_frame attribute if it doesn't exist
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return []

        # Calculate frame difference to detect motion
        diff = cv2.absdiff(self.previous_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh_diff = cv2.threshold(blur_diff, 20, 255, cv2.THRESH_BINARY)

        # Update detection tracking dict with age management
        current_time = time.time()
        outdated_keys = []
        for pos, (last_time, count) in self.consecutive_detections.items():
            if current_time - last_time > 5.0:  # Remove entries older than 5 seconds
                outdated_keys.append(pos)
        for key in outdated_keys:
            del self.consecutive_detections[key]

        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > self.confidence_threshold and cls == self.person_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1
                    h = y2 - y1
                    candidate = (x1, y1, w, h)

                    if self._inside_roi(candidate):
                        # Create mask for the detected box area
                        mask = np.zeros_like(thresh_diff)
                        mask[y1:y2, x1:x2] = 1

                        # Calculate the motion in the box area
                        motion_pixels = cv2.countNonZero(thresh_diff * mask)
                        area = w * h
                        motion_percentage = (motion_pixels / area) * 100

                        # Position key for tracking (based on grid cell)
                        grid_x, grid_y = x1 // 50, y1 // 50
                        pos_key = (grid_x, grid_y)

                        # Update tracking for this position
                        if motion_percentage > self.movement_threshold:
                            if pos_key in self.consecutive_detections:
                                _, count = self.consecutive_detections[pos_key]
                                self.consecutive_detections[pos_key] = (current_time, count + 1)
                            else:
                                self.consecutive_detections[pos_key] = (current_time, 1)

                            # Only add person if detected consistently or with significant motion
                            if self.consecutive_detections[pos_key][
                                1] >= 2 or motion_percentage > self.movement_threshold * 2:
                                persons.append(candidate)
                        else:
                            # Reduce confidence for this position
                            if pos_key in self.consecutive_detections:
                                _, count = self.consecutive_detections[pos_key]
                                self.consecutive_detections[pos_key] = (current_time, max(0, count - 1))

        # Update frame history
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)
        self.previous_frame = frame.copy()

        return persons

    def draw_persons(self, frame, persons):
        for (x, y, w, h) in persons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw ROI rectangle if defined
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        return frame

    def export_image(self, frame, camera_name):
        # Ensure export directory exists.
        export_dir = "exports"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        timestamp = int(time.time())
        filename = f"{export_dir}/{camera_name}_person_{timestamp}.png"
        if cv2.imwrite(filename, frame):
            print(f"Exported positive result to {filename}")
        else:
            print(f"Failed to export image to {filename}")


def open_stream(rtsp_url, width=640, height=480):
    # Force FFmpeg to use UDP transport for RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url)

    # Set the capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    time.sleep(2)  # Give the stream time to initialize
    return cap


def monitor_camera(camera_url, window_name, roi, stop_event, display_frames, display_lock):
    detection = PersonDetection(roi=roi)

    # Use a different capture method based on the window name
    if "(HIK)" in window_name:
        print(f"{window_name}: Using HIK Vision capture method")
        cap = open_stream(camera_url)
    else:
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        target_fps = 30
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    target_fps = 30
    frame_time = 1 / target_fps
    frame_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"{window_name}: Unable to capture frame.")
            break

        persons = detection.detect_persons(frame)
        if persons:
            detection.notifier.speak("There is a person")
        # detection.export_image(frame, window_name)

        processed_frame = detection.draw_persons(frame, persons)
        cv2.putText(processed_frame, window_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with display_lock:
            display_frames[window_name] = processed_frame

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"{window_name} FPS: {fps:.2f}")

        processing_time = time.time() - loop_start
        delay = max(0, frame_time - processing_time)
        time.sleep(delay)

    cap.release()


def combine_frames(frames):
    if not frames:
        return None
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w, channels = frames[0].shape
    resized_frames = [cv2.resize(frame, (w, h)) for frame in frames]

    grid_rows = []
    for i in range(rows):
        row_frames = []
        for j in range(cols):
            idx = i * cols + j
            if idx < len(resized_frames):
                row_frames.append(resized_frames[idx])
            else:
                row_frames.append(np.zeros((h, w, channels), dtype=np.uint8))
        row = cv2.hconcat(row_frames)
        grid_rows.append(row)
    combined_frame = cv2.vconcat(grid_rows)
    return combined_frame


def main():
    cameras = [
        {
            'url': 'rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1',
            'name': 'UVK Gate',
            'roi': (250, 0, 450, 400)
        },
        {
            'url': 'rtsp://admin:L268C6B7@d5030edfff7a.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1',
            'name': 'UVK Parking',
            'roi': (100, 50, 400, 400)
        },
        {
            'url': 'rtsp://admin:L201353B@hcr086zs3b5.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1',
            'name': 'LBB Rooftop',
            'roi': (150, 20, 500, 500)
        },
        {
            'url': 'rtsp://user1:1234abcd@115.79.213.124:10554/streaming/channels/502',
            'name': '(HIK) PNA Tennis 2',
            'roi': (50, 200, 800, 250)
        }
    ]

    threads = []
    for cam in cameras:
        t = threading.Thread(
            target=monitor_camera,
            args=(cam['url'], cam['name'], cam['roi'], stop_event, display_frames, display_lock)
        )
        t.start()
        threads.append(t)

    while not stop_event.is_set():
        with display_lock:
            frames = list(display_frames.values())
        if frames:
            combined_frame = combine_frames(frames)
            if combined_frame is not None:
                cv2.imshow("Combined Cameras", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
