import cv2
import time
import threading
from datetime import datetime, date
import numpy as np

import database
from face_recognition import FaceEngine

class VideoStream:
    """Threaded camera stream for low-latency real-time video capture."""
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.is_running:
            return True

        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
        if not self.cap.isOpened():
            # Try without CAP_DSHOW
            self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            print(f"[VideoStream] Unable to open camera source {self.src}")
            return False

        # Set optimal resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


class AttendanceSystem:
    def __init__(self):
        self.face_engine = FaceEngine()
        self.video_stream = VideoStream()
        
        # In-memory enrolled embeddings: list of (roll_number, name, emb_bytes)
        self.enrolled_students = []
        self.reload_enrolled_students()

        # Cooldown tracker to avoid re-triggering database writes repeatedly in consecutive frames
        # dict of {roll_number: timestamp}
        self.cooldown_tracker = {}
        self.cooldown_seconds = 15.0

        # Latest event for UI feed
        self.latest_event = {
            'type': 'idle',
            'message': 'System Ready',
            'timestamp': datetime.now().strftime('%I:%M:%S %p'),
            'student': None
        }

        # Real-time metrics
        self.fps = 0.0
        self._prev_time = time.time()
        self._frame_count = 0

    def reload_enrolled_students(self):
        """Refreshes the in-memory cache of enrolled embeddings from SQLite."""
        self.enrolled_students = database.get_all_embeddings()
        print(f"[AttendanceSystem] Loaded {len(self.enrolled_students)} enrolled students into recognition memory.")

    def start_camera(self):
        return self.video_stream.start()

    def stop_camera(self):
        self.video_stream.stop()

    def process_frame(self, frame):
        """
        Main pipeline:
        Frame → Face Detection → Face Recognition → Attendance DB Check → HUD Visualization
        """
        if frame is None:
            return None

        # Compute FPS
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._prev_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._prev_time = now

        faces = self.face_engine.detect_faces(frame)

        for face in faces:
            bbox = face['bbox']
            raw_face = face['raw_face']
            
            # Anti-spoofing / Liveness
            is_live, liveness_msg = self.face_engine.check_liveness(frame, bbox)
            if not is_live:
                self.face_engine.draw_face_hud(
                    frame, bbox, "Suspicious / Blur",
                    status_text=liveness_msg,
                    color=(0, 165, 255) # Orange warning
                )
                continue

            # Extract 128-D Deep Feature Embedding
            feature = self.face_engine.extract_feature(frame, raw_face)
            
            # Match with registered students
            roll, name, confidence, is_match = self.face_engine.match_face(feature, self.enrolled_students)

            if is_match and roll:
                # Check cooldown to avoid hammering the database
                last_seen = self.cooldown_tracker.get(roll, 0)
                current_time = time.time()

                if current_time - last_seen > self.cooldown_seconds:
                    # Mark attendance in database
                    success, msg, student_info = database.mark_attendance(roll, confidence=confidence)
                    self.cooldown_tracker[roll] = current_time

                    self.latest_event = {
                        'type': 'success' if success else 'info',
                        'message': msg,
                        'timestamp': datetime.now().strftime('%I:%M:%S %p'),
                        'student': student_info
                    }

                # Color coding based on whether already present or just marked
                # Check if today's attendance already exists
                today_attendance = database.get_today_attendance()
                already_present = any(att['roll_number'] == roll for att in today_attendance)

                if already_present:
                    color = (255, 180, 50) # Cyan/Blue for Already Present
                    status_str = f"Present today ({confidence:.0f}%)"
                else:
                    color = (50, 220, 50)  # Green for Just Verified
                    status_str = f"Verified ({confidence:.0f}%)"

                self.face_engine.draw_face_hud(
                    frame, bbox, name, roll_number=roll,
                    status_text=status_str,
                    color=color
                )
            else:
                # Unknown Face
                self.face_engine.draw_face_hud(
                    frame, bbox, "Unknown Person",
                    status_text=f"No match ({confidence:.0f}%)",
                    color=(50, 50, 230) # Red
                )

        # Header HUD
        hud_bg = frame[:40, :].copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 24, 33), -1)
        cv2.putText(frame, f"AI ATTENDANCE SYSTEM  |  FPS: {self.fps}  |  Faces: {len(faces)}", 
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)
        
        # System clock
        clock_str = datetime.now().strftime("%Y-%m-%d  %I:%M:%S %p")
        (tw, _), _ = cv2.getTextSize(clock_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, clock_str, (frame.shape[1] - tw - 15, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def generate_video_stream(self):
        """Yields JPEG multipart frames for Flask /video_feed."""
        # Ensure camera is started
        if not self.video_stream.is_running:
            self.start_camera()

        while True:
            frame = self.video_stream.read()
            if frame is None:
                # If camera is not ready, generate placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to camera / Camera offline...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                time.sleep(0.1)
                continue

            processed = self.process_frame(frame)
            ret, jpeg = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Global singleton
attendance_system = AttendanceSystem()
