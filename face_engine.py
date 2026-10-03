import cv2
import numpy as np
import os
import time

class FaceEngine:
    def __init__(self, 
                 yunet_path='models/face_detection_yunet.onnx', 
                 sface_path='models/face_recognition_sface.onnx',
                 score_threshold=0.5,
                 nms_threshold=0.3,
                 cosine_threshold=0.363):
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.yunet_path = os.path.join(base_dir, yunet_path)
        self.sface_path = os.path.join(base_dir, sface_path)
        self.cascade_path = os.path.join(base_dir, 'models', 'haarcascade_frontalface_default.xml')

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.cosine_threshold = cosine_threshold

        self.detector = None
        self.recognizer = None
        self.face_cascade = None
        self._input_size = None

        self._init_models()

    def _init_models(self):
        """Initialize YuNet and SFace models, with Haar cascade fallback."""
        if os.path.exists(self.yunet_path) and os.path.exists(self.sface_path):
            try:
                # Initial dummy size (320, 320)
                self.detector = cv2.FaceDetectorYN.create(
                    self.yunet_path,
                    "",
                    (320, 320),
                    self.score_threshold,
                    self.nms_threshold,
                    top_k=5000
                )
                self._input_size = (320, 320)
                self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")
            except Exception as e:
                print(f"[FaceEngine] Error loading DNN models: {e}")

        # Haar cascade fallback
        if os.path.exists(self.cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame):
        """
        Detects faces in the given BGR frame.
        Returns: list of dicts: [{'bbox': (x, y, w, h), 'landmarks': [...], 'raw_face': array, 'score': float}]
        """
        if frame is None or self.detector is None:
            return []

        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None:
            return []

        results = []
        for face in raw_faces:
            x, y, fw, fh = map(int, face[0:4])
            score = float(face[-1])
            # Ensure within bounds
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw < 30 or fh < 30:
                continue

            landmarks = face[4:14].reshape(5, 2)
            results.append({
                'bbox': (x, y, fw, fh),
                'landmarks': landmarks,
                'raw_face': face,
                'score': score
            })
        return results

    def extract_feature(self, frame, raw_face):
        """
        Aligns the face and computes a 128-dimensional embedding vector.
        """
        if self.recognizer is None or frame is None or raw_face is None:
            return None

        try:
            aligned_face = self.recognizer.alignCrop(frame, raw_face)
            feature = self.recognizer.feature(aligned_face)
            # Normalize vector
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"[FaceEngine] Alignment / Feature extraction error: {e}")
            return None

    def match_face(self, feature, enrolled_students):
        """
        Compares `feature` with all enrolled student embeddings.
        enrolled_students: list of (roll_number, name, embedding_bytes)
        Returns: (roll_number or None, name or 'Unknown', similarity_score: float, is_match: bool)
        """
        if feature is None or not enrolled_students:
            return None, "Unknown", 0.0, False

        best_roll = None
        best_name = "Unknown"
        max_score = -1.0

        for roll, name, emb_bytes in enrolled_students:
            if emb_bytes is None:
                continue
            
            stored_emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, -1)
            score = self.recognizer.match(feature, stored_emb, cv2.FaceRecognizerSF_FR_COSINE)
            
            if score > max_score:
                max_score = score
                best_roll = roll
                best_name = name

        is_match = max_score >= self.cosine_threshold
        # Convert cosine score to percentage for readability (typically 0.36 to 0.8+ range)
        # Cosine score is [-1, 1], but for matching faces it is typically [0.36, 0.90]
        confidence_pct = min(100.0, max(0.0, (max_score + 1.0) / 2.0 * 100.0))

        if is_match:
            return best_roll, best_name, confidence_pct, True
        else:
            return None, "Unknown", confidence_pct, False

    def check_liveness(self, frame, bbox):
        """
        Performs light liveness / anti-spoofing heuristic checks:
        1. Laplacian variance (blur / paper print detection)
        2. Reasonable size
        Returns: (is_live: bool, reason: str)
        """
        x, y, w, h = bbox
        if w < 60 or h < 60:
            return False, "Face too small / far from camera"

        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return False, "Invalid face area"

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Extremely low variance indicates an out-of-focus image or soft photo paper
        if laplacian_var < 35:
            return False, f"Image blurred/spoof suspected (Var: {laplacian_var:.1f})"

        return True, "Live face verified"

    def draw_face_hud(self, frame, bbox, name, roll_number=None, status_text=None, color=(0, 255, 0)):
        """
        Draws an aesthetic bounding box and modern HUD overlay around the face.
        """
        x, y, w, h = bbox
        corner_len = int(min(w, h) * 0.25)
        thickness = 2
        corner_thickness = 4

        # Subtle semi-transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Highlighted corners
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thickness)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thickness)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thickness)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thickness)

        # Label tag
        label = name
        if roll_number:
            label += f" | {roll_number}"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        text_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        tag_y = max(y - 10, label_h + 12)
        cv2.rectangle(frame, (x, tag_y - label_h - 8), (x + label_w + 14, tag_y + 4), color, -1)
        cv2.putText(frame, label, (x + 7, tag_y - 3), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        # Status subtitle if present
        if status_text:
            sub_y = y + h + 20
            sub_scale = 0.45
            (sw, sh), _ = cv2.getTextSize(status_text, font, sub_scale, 1)
            cv2.rectangle(frame, (x, sub_y - sh - 4), (x + sw + 10, sub_y + 4), (30, 30, 30), -1)
            cv2.putText(frame, status_text, (x + 5, sub_y - 2), font, sub_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
