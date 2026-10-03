"""
AI Real-Time Attendance System - Face Recognition Module
Contains Face Detection (OpenCV YuNet DNN), Face Recognition (OpenCV SFace 128-D Embeddings),
Liveness/Anti-Spoofing checks, and HUD drawing.
"""

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
        """Initializes Deep Neural Network models (YuNet detector & SFace recognizer)."""
        if os.path.exists(self.yunet_path) and os.path.exists(self.sface_path):
            try:
                # Default input size (320, 320)
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
                print("[FaceEngine] OpenCV YuNet & SFace models initialized successfully.")
            except Exception as e:
                print(f"[FaceEngine] Error loading DNN models: {e}")

        # Haar cascade fallback
        if os.path.exists(self.cascade_path):
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, frame, fallback_to_cascade=True):
        """
        Detects faces in BGR frame using YuNet, with Haar Cascade fallback.
        Returns: list of dicts: [{'bbox': (x, y, w, h), 'landmarks': 5x2 array, 'raw_face': array, 'score': float}]
        """
        if frame is None or self.detector is None:
            return []

        h, w = frame.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, raw_faces = self.detector.detect(frame)
        results = []
        if raw_faces is not None:
            for face in raw_faces:
                x, y, fw, fh = map(int, face[0:4])
                score = float(face[-1])
                # Bound coordinates
                x = max(0, x)
                y = max(0, y)
                fw = min(fw, w - x)
                fh = min(fh, h - y)

                if fw < 20 or fh < 20:
                    continue

                landmarks = face[4:14].reshape(5, 2)
                results.append({
                    'bbox': (x, y, fw, fh),
                    'landmarks': landmarks,
                    'raw_face': face,
                    'score': score
                })

        if results:
            return results

        # Fallback to Haar Cascade if YuNet found no faces
        if fallback_to_cascade and self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30))
            for (cx, cy, cw, ch) in cascade_faces:
                re_x, re_y = cx + cw * 0.3, cy + ch * 0.35
                le_x, le_y = cx + cw * 0.7, cy + ch * 0.35
                nt_x, nt_y = cx + cw * 0.5, cy + ch * 0.55
                rc_x, rc_y = cx + cw * 0.35, cy + ch * 0.75
                lc_x, lc_y = cx + cw * 0.65, cy + ch * 0.75
                sim_face = np.array([cx, cy, cw, ch, re_x, re_y, le_x, le_y, nt_x, nt_y, rc_x, rc_y, lc_x, lc_y, 0.85], dtype=np.float32)
                results.append({
                    'bbox': (int(cx), int(cy), int(cw), int(ch)),
                    'landmarks': sim_face[4:14].reshape(5, 2),
                    'raw_face': sim_face,
                    'score': 0.85
                })

        return results

    def extract_feature(self, frame, raw_face):
        """
        Aligns the detected face crop and computes a 128-dimensional embedding vector.
        """
        if self.recognizer is None or frame is None or raw_face is None:
            return None

        try:
            aligned_face = self.recognizer.alignCrop(frame, raw_face)
            feature = self.recognizer.feature(aligned_face)
            # L2 Normalize vector
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"[FaceEngine] Alignment / Feature extraction error: {e}")
            return None

    def extract_multi_sample_embedding(self, images_list):
        """
        Enrolls multiple face images of a student, extracts 128-D vectors for each,
        and averages them into a robust composite embedding.
        """
        embeddings = []
        for img in images_list:
            if img is None:
                continue
            faces = self.detect_faces(img)
            if faces:
                emb = self.extract_feature(img, faces[0]['raw_face'])
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            return None

        # Mean vector
        composite = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite = composite / norm
        return composite

    def match_face(self, feature, enrolled_students):
        """
        Compares `feature` with all enrolled student embeddings using Cosine Similarity.
        enrolled_students: list of (roll_number, name, embedding_bytes)
        Returns: (roll_number or None, name or 'Unknown', confidence_pct: float, is_match: bool)
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
        # Convert cosine score to a calibrated confidence percentage
        confidence_pct = min(100.0, max(0.0, (max_score + 1.0) / 2.0 * 100.0))

        if is_match:
            return best_roll, best_name, confidence_pct, True
        else:
            return None, "Unknown", confidence_pct, False

    def check_liveness(self, frame, bbox):
        """
        Anti-spoofing check using Laplacian variance to detect blurry / printed photos / screens.
        """
        x, y, w, h = bbox
        if w < 50 or h < 50:
            return False, "Face too far from camera"

        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return False, "Invalid face area"

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 35:
            return False, f"Spoof/Blur suspected (Var: {laplacian_var:.1f})"

        return True, "Live face verified"

    def draw_face_hud(self, frame, bbox, name, roll_number=None, status_text=None, color=(0, 255, 0)):
        """
        Renders HUD brackets, identification badge, and status subtitles.
        """
        x, y, w, h = bbox
        corner_len = int(min(w, h) * 0.25)
        thickness = 2
        corner_thickness = 4

        # Transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Corners
        cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thickness)
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thickness)

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

        if status_text:
            sub_y = y + h + 20
            sub_scale = 0.45
            (sw, sh), _ = cv2.getTextSize(status_text, font, sub_scale, 1)
            cv2.rectangle(frame, (x, sub_y - sh - 4), (x + sw + 10, sub_y + 4), (30, 30, 30), -1)
            cv2.putText(frame, status_text, (x + 5, sub_y - 2), font, sub_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

# Alias for compatibility
FaceRecognition = FaceEngine
