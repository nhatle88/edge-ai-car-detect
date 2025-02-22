import threading
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time
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
    def __init__(self, confidence_threshold=0.4, roi=None):
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
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > self.confidence_threshold and cls == self.person_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    candidate = (int(x1), int(y1), int(w), int(h))
                    if self._inside_roi(candidate):
                        persons.append(candidate)
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


def monitor_camera(camera_url, window_name, roi, stop_event, display_frames, display_lock):
    detection = PersonDetection(roi=roi)
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            detection.export_image(frame, window_name)

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
            'url': 'rtsp://admin:L268C6B7@d5030edfff7a.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1',
            'name': 'UVK Parking',
            'roi': (100, 50, 400, 400)
        },
        {
            'url': 'rtsp://admin:L2EC70CF@d5030edfff7a.sn.mynetname.net:554/cam/realmonitor?channel=1&subtype=1',
            'name': 'UVK Gate',
            'roi': (250, 0, 450, 400)
        },
        {
            'url': 'rtsp://admin:L201353B@hcr086zs3b5.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1',
            'name': 'LBB Rooftop',
            'roi': (150, 20, 500, 500)
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