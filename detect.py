import torch
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import subprocess

class Notifier:
    def __init__(self):
        pass

    def speak(self, text):
        subprocess.run(['say', text])

class VehicleTracker:
    def __init__(self, confidence_threshold=0.4, max_disappeared=30*10):
        # Initialize YOLO model with GPU support
        self.notifier = Notifier()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.notifier.speak("Detection Initiated")
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # or 'yolov8n.pt' for less accuracy but faster inference
        self.model.to(self.device)
        
        # Tracking parameters
        self.confidence_threshold = confidence_threshold
        self.max_disappeared = max_disappeared
        self.next_vehicle_id = 0
        self.vehicles = {}
        self.vehicle_history = defaultdict(list)
        
        # Valid vehicle classes in YOLO v8
        self.vehicle_classes = [2, 5, 7]  # car, bus, truck in YOLOv8
        
    def process_frame(self, frame, target_fps=10):
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(frame_rgb, verbose=False)
        
        # Process detections
        current_vehicles = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf > self.confidence_threshold and cls in self.vehicle_classes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    current_vehicles.append((int(x1), int(y1), int(w), int(h)))
        
        # Update tracking
        self.update_tracking(current_vehicles)
        
        # Draw results
        for vehicle_id, vehicle_info in self.vehicles.items():
            if vehicle_info["disappeared"] == 0:
                x, y, w, h = vehicle_info["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add status text
                status = self.get_vehicle_status(vehicle_id)
                cv2.putText(frame, f"ID: {vehicle_id} ({status})", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        return frame
    
    def update_tracking(self, current_vehicles):
        # Mark all existing vehicles as disappeared initially
        for vehicle_id in self.vehicles:
            self.vehicles[vehicle_id]["disappeared"] += 1
        
        # Update or add new vehicles
        for box in current_vehicles:
            matched = False
            for vehicle_id, vehicle_info in self.vehicles.items():
                if self.calculate_overlap(box, vehicle_info["box"]) > 0.3:
                    self.vehicles[vehicle_id]["box"] = box
                    self.vehicles[vehicle_id]["disappeared"] = 0
                    self.vehicle_history[vehicle_id].append(time.time())
                    matched = True
                    break
            
            if not matched:
                self.notifier.speak("Vehicle Arriving")
                self.vehicles[self.next_vehicle_id] = {
                    "box": box,
                    "disappeared": 0
                }
                self.vehicle_history[self.next_vehicle_id].append(time.time())
                self.next_vehicle_id += 1
        
        # Remove vehicles that have disappeared for too long
        for vehicle_id in list(self.vehicles.keys()):
            if self.vehicles[vehicle_id]["disappeared"] > self.max_disappeared:
                self.notifier.speak("Vehicle Leaving")
                del self.vehicles[vehicle_id]
    
    # Calculate IoU between two boxes (Intersection over Union)

    def calculate_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # Get vehicle status based on time present
    
    def get_vehicle_status(self, vehicle_id):
        if vehicle_id not in self.vehicle_history:
            return "Unknown"
        
        timestamps = self.vehicle_history[vehicle_id]
        if len(timestamps) < 2:
            return "Arriving"
        
        time_present = timestamps[-1] - timestamps[0]
        if time_present < 3:
            return "Arriving"
        else:
            return "Present"

def main():
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Access RTSP stream
    cap = cv2.VideoCapture('rtsp://admin:L268C6B7@d5030edfff7a.sn.mynetname.net:556/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
    
    # Set buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Target FPS and frame timing
    target_fps = 30  # Increased since we're using GPU
    frame_time = 1/target_fps
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    
    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = tracker.process_frame(frame)
        
        # Display the output
        cv2.imshow('Vehicle Detection', processed_frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.2f}")
        
        # Maintain target FPS
        processing_time = time.time() - loop_start
        delay = max(1, int((frame_time - processing_time) * 1000))
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":   
    main()
