"""
VisionAttend - Standalone Desktop Real-Time Attendance Runner
Run this script to launch a native OpenCV desktop window with webcam feed,
bounding boxes, real-time face recognition, and automatic database logging.
"""

import cv2
import time
from attendance import attendance_system
import database

def main():
    print("=" * 60)
    print(" VisionAttend - AI Real-Time Attendance System (Desktop)")
    print("=" * 60)
    print("Initializing camera feed... Press 'q' or 'ESC' to exit.")

    # Reload enrolled students from database
    attendance_system.reload_enrolled_students()
    enrolled_count = len(attendance_system.enrolled_students)
    print(f"Loaded {enrolled_count} registered students.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam. Please verify camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "AI Real-Time Attendance System (OpenCV + DeepFace/SFace)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Warning] Frame capture failed.")
            time.sleep(0.05)
            continue

        # Process frame through full detection + recognition + attendance pipeline
        processed = attendance_system.process_frame(frame)

        # Show desktop window
        cv2.imshow(window_name, processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            print("\nExiting attendance monitor...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. System stopped.")

if __name__ == '__main__':
    main()
