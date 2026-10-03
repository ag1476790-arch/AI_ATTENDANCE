import os
import io
import re
import csv
import base64
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, Response, request, 
    jsonify, send_file, send_from_directory, 
    session, redirect, url_for
)

import database
from attendance import attendance_system

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise RuntimeError(
        "Set the FLASK_SECRET_KEY environment variable before starting the web app."
    )
if not database.has_admin_accounts():
    raise RuntimeError(
        "No admin account exists. Set ADMIN_USERNAME and ADMIN_PASSWORD, then restart."
    )

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)
frame_processing_lock = Lock()

# ----------------------------------------------------
# Static Image Serving for Dataset
# ----------------------------------------------------
@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    return send_from_directory(DATASET_DIR, filename)

# ----------------------------------------------------
# Authentication Routes
# ----------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if database.verify_admin(username, password):
            session['admin_user'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password. Please try again."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('admin_user', None)
    return redirect(url_for('login'))

# ----------------------------------------------------
# Pages
# ----------------------------------------------------
@app.route('/')
def dashboard():
    stats = database.get_dashboard_stats()
    today_records = database.get_today_attendance()
    return render_template(
        'dashboard.html',
        active_page='dashboard',
        stats=stats,
        today_records=today_records,
        admin_user=session.get('admin_user', 'Admin')
    )

@app.route('/students')
def students():
    student_list = database.get_all_students()
    return render_template(
        'students.html',
        active_page='students',
        students=student_list,
        admin_user=session.get('admin_user', 'Admin')
    )

@app.route('/reports')
def reports():
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    roll_number = request.args.get('roll_number', '')
    department = request.args.get('department', '')

    records = database.get_attendance_reports(
        start_date=start_date or None,
        end_date=end_date or None,
        roll_number=roll_number or None,
        department=department or None
    )
    student_analytics = database.get_student_analytics()

    filter_params = {
        'start_date': start_date,
        'end_date': end_date,
        'roll_number': roll_number,
        'department': department
    }

    return render_template(
        'reports.html',
        active_page='reports',
        records=records,
        student_analytics=student_analytics,
        filter_params=filter_params,
        admin_user=session.get('admin_user', 'Admin')
    )

@app.route('/theory')
def theory():
    return render_template('theory.html', active_page='theory', admin_user=session.get('admin_user', 'Admin'))

# ----------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------
@app.route('/api/live_status')
def api_live_status():
    """Poll endpoint providing real-time data to dashboard."""
    stats = database.get_dashboard_stats()
    today_attendance = database.get_today_attendance()
    return jsonify({
        'fps': attendance_system.fps,
        'event': attendance_system.latest_event,
        'stats': stats,
        'today_attendance': today_attendance
    })

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """Process a JPEG frame captured by the user's browser camera."""
    if request.content_length and request.content_length > 1_500_000:
        return jsonify({'status': False, 'message': 'The camera frame is too large.'}), 413

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('image'), str):
        return jsonify({'status': False, 'message': 'A base64 JPEG image is required.'}), 400

    image_data = data['image']
    if image_data.startswith('data:'):
        header, separator, image_data = image_data.partition(',')
        if not separator or header.lower() != 'data:image/jpeg;base64':
            return jsonify({'status': False, 'message': 'Only base64 JPEG frames are accepted.'}), 400

    if not image_data or len(image_data) > 1_000_000:
        return jsonify({'status': False, 'message': 'The camera frame is empty or too large.'}), 413

    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (ValueError, base64.binascii.Error):
        return jsonify({'status': False, 'message': 'The camera frame is not valid base64 data.'}), 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'status': False, 'message': 'The camera frame is not a valid JPEG image.'}), 400

    height, width = frame.shape[:2]
    if width > 1280 or height > 960:
        return jsonify({'status': False, 'message': 'Camera frames must be at most 1280x960.'}), 400

    try:
        with frame_processing_lock:
            attendance_system.process_frame(frame)
    except Exception:
        app.logger.exception("Failed to process a browser camera frame")
        return jsonify({'status': False, 'message': 'The server could not process the camera frame.'}), 500

    return jsonify({'status': True, 'fps': attendance_system.fps})

@app.route('/api/register_student', methods=['POST'])
def api_register_student():
    """
    Registers a student and extracts their 128-D facial vector embedding.
    Accepts JSON with: name, roll_number, department, email, image OR images list.
    Supports multi-image enrollment for enhanced biometric accuracy.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': False, 'message': 'Invalid JSON request payload.'}), 400

        name = data.get('name', '').strip()
        roll = data.get('roll_number', '').strip().upper()
        department = data.get('department', '').strip()
        email = data.get('email', '').strip()
        image_data = data.get('image', '')
        images_list_data = data.get('images', [])

        if not name or not roll or not department:
            return jsonify({'status': False, 'message': 'Name, Roll Number, and Department are required.'}), 400

        # Collect raw images
        raw_images = []
        if images_list_data and isinstance(images_list_data, list):
            for img_b64 in images_list_data:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                dec = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
                if dec is not None:
                    raw_images.append(dec)
        elif image_data:
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            dec = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
            if dec is not None:
                raw_images.append(dec)

        if not raw_images:
            return jsonify({'status': False, 'message': 'Please provide face photo(s) for biometric registration.'}), 400

        engine = attendance_system.face_engine

        # If multiple images provided, use multi-sample composite embedding
        if len(raw_images) > 1:
            embedding = engine.extract_multi_sample_embedding(raw_images)
            primary_img = raw_images[0]
            faces = engine.detect_faces(primary_img)
            target_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3]) if faces else None
        else:
            primary_img = raw_images[0]
            faces = engine.detect_faces(primary_img)
            if not faces:
                # Retry with slightly resized frame if very high or low res
                h, w = primary_img.shape[:2]
                resized = cv2.resize(primary_img, (640, int(640 * h / w)))
                faces = engine.detect_faces(resized)
                if faces:
                    primary_img = resized

            if not faces:
                # Portrait crop fallback for close-up selfies
                h, w = primary_img.shape[:2]
                cw, ch = int(w * 0.7), int(h * 0.7)
                cx, cy = int((w - cw) / 2), int((h - ch) / 2)
                sim_face = np.array([cx, cy, cw, ch, cx + cw * 0.3, cy + ch * 0.35, cx + cw * 0.7, cy + ch * 0.35, cx + cw * 0.5, cy + ch * 0.55, cx + cw * 0.35, cy + ch * 0.75, cx + cw * 0.65, cy + ch * 0.75, 0.75], dtype=np.float32)
                target_face = {
                    'bbox': (cx, cy, cw, ch),
                    'landmarks': sim_face[4:14].reshape(5, 2),
                    'raw_face': sim_face,
                    'score': 0.75
                }
                faces = [target_face]

            # Select the largest, most prominent face
            target_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
            embedding = engine.extract_feature(primary_img, target_face['raw_face'])

        if embedding is None:
            return jsonify({'status': False, 'message': 'Could not extract valid facial embedding vector. Please try another shot.'}), 400

        # Crop and save representative student photo
        if target_face:
            x, y, w, h = target_face['bbox']
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            img_h, img_w = primary_img.shape[:2]
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(img_w, x + w + pad_x)
            crop_y2 = min(img_h, y + h + pad_y)
            face_crop = primary_img[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            face_crop = primary_img

        photo_filename = f"{roll}.jpg"
        photo_path = os.path.join(DATASET_DIR, photo_filename)
        cv2.imwrite(photo_path, face_crop if face_crop.size > 0 else primary_img)

        # Save to database
        emb_bytes = embedding.astype(np.float32).tobytes()
        success, msg = database.add_student(
            roll_number=roll,
            name=name,
            department=department,
            email=email,
            embedding=emb_bytes,
            photo_path=photo_path
        )

        if not success:
            if "already exists" in msg.lower():
                database.update_student_embedding(roll, emb_bytes, photo_path)
                attendance_system.reload_enrolled_students()
                return jsonify({
                    'status': True,
                    'message': f'Student {name} ({roll}) biometrics updated successfully!'
                })
            return jsonify({'status': False, 'message': msg}), 400

        # Reload recognition cache
        attendance_system.reload_enrolled_students()

        return jsonify({
            'status': True,
            'message': f'Student {name} ({roll}) successfully registered with biometric vectors!'
        })

    except Exception as e:
        return jsonify({'status': False, 'message': f'Registration error: {str(e)}'}), 500

@app.route('/api/delete_student/<roll_number>', methods=['DELETE'])
def api_delete_student(roll_number):
    try:
        roll = roll_number.strip().upper()
        deleted = database.delete_student(roll)
        if deleted:
            photo_path = os.path.join(DATASET_DIR, f"{roll}.jpg")
            if os.path.exists(photo_path):
                os.remove(photo_path)
            
            attendance_system.reload_enrolled_students()
            return jsonify({'status': True, 'message': f'Student {roll} deleted successfully.'})
        else:
            return jsonify({'status': False, 'message': f'Student {roll} not found.'}), 404
    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500

@app.route('/export_csv')
def export_csv():
    """Exports attendance records to CSV."""
    start_date = request.args.get('start_date') or None
    end_date = request.args.get('end_date') or None
    roll_number = request.args.get('roll_number') or None
    department = request.args.get('department') or None

    records = database.get_attendance_reports(
        start_date=start_date,
        end_date=end_date,
        roll_number=roll_number,
        department=department
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Roll Number', 'Student Name', 'Department', 'Date', 'Time', 'Status', 'Confidence (%)'])

    for r in records:
        writer.writerow([r['id'], r['roll_number'], r['name'], r['department'], r['date'], r['time'], r['status'], r['confidence']])

    filename = f"Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@app.route('/export_excel')
def export_excel():
    """Exports attendance records to styled Excel workbook (.xlsx)."""
    start_date = request.args.get('start_date') or None
    end_date = request.args.get('end_date') or None
    roll_number = request.args.get('roll_number') or None
    department = request.args.get('department') or None

    records = database.get_attendance_reports(
        start_date=start_date,
        end_date=end_date,
        roll_number=roll_number,
        department=department
    )

    df = pd.DataFrame(records)
    if not df.empty:
        # Rename columns for professional presentation
        rename_map = {
            'id': 'Record ID',
            'roll_number': 'Roll Number',
            'name': 'Student Name',
            'department': 'Department',
            'date': 'Attendance Date',
            'time': 'Time Logged',
            'status': 'Status',
            'confidence': 'AI Confidence (%)'
        }
        df = df[[col for col in rename_map.keys() if col in df.columns]]
        df = df.rename(columns=rename_map)
    else:
        df = pd.DataFrame(columns=['Record ID', 'Roll Number', 'Student Name', 'Department', 'Attendance Date', 'Time Logged', 'Status', 'AI Confidence (%)'])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Attendance Logs', index=False)
    excel_buffer.seek(0)

    filename = f"Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    print("=" * 65)
    print(" VisionAttend - AI Real-Time Facial Attendance System")
    print(" Dashboard active on: http://127.0.0.1:5000")
    print(" Set ADMIN_USERNAME and ADMIN_PASSWORD before first launch.")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
