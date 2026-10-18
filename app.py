import os
import io
import re
import csv
import base64
import functools
import urllib.parse
from datetime import datetime
from threading import Lock

import cv2
import requests
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
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'visionattend-secret-key-2026-production')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

# OAuth Credentials from environment
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '').strip()
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '').strip()
GITHUB_CLIENT_ID = os.environ.get('GITHUB_CLIENT_ID', '').strip()
GITHUB_CLIENT_SECRET = os.environ.get('GITHUB_CLIENT_SECRET', '').strip()

# Firebase Authentication configuration
FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID', 'ai-attendance-14ab6').strip()
FIREBASE_AUTH_DOMAIN = os.environ.get('FIREBASE_AUTH_DOMAIN', 'ai-attendance-14ab6.firebaseapp.com').strip()
FIREBASE_API_KEY = os.environ.get('FIREBASE_API_KEY', '').strip()

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)
frame_processing_lock = Lock()

def is_admin():
    """Returns True if current session holds administrator privileges."""
    return session.get('role') == 'admin' or bool(session.get('admin_user'))

def admin_required(f):
    """Restricts route access strictly to authenticated Administrators."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            if request.path.startswith('/api/'):
                return jsonify({
                    'status': False,
                    'message': 'Access Denied: Only administrators are authorized to create, update, or delete student records.'
                }), 403
            return redirect(url_for('login', role='admin', error="Administrator credentials required to perform this action."))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_context():
    return {
        'user_session': session.get('user', {}),
        'is_admin': is_admin(),
        'current_role': session.get('role', 'admin' if session.get('admin_user') else None),
        'firebase_project_id': FIREBASE_PROJECT_ID,
        'firebase_auth_domain': FIREBASE_AUTH_DOMAIN,
        'firebase_api_key': FIREBASE_API_KEY,
    }

# ----------------------------------------------------
# Static Image Serving for Dataset
# ----------------------------------------------------
@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    return send_from_directory(DATASET_DIR, filename)

# ----------------------------------------------------
# Authentication Routes (Student & Admin, Google & GitHub)
# ----------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
@app.route('/auth', methods=['GET'])
def login():
    initial_mode = request.args.get('mode', 'login').lower()
    if initial_mode not in ('login', 'register'):
        initial_mode = 'login'
    initial_role = request.args.get('role', 'student').lower()
    if initial_role not in ('student', 'admin'):
        initial_role = 'student'
    error = request.args.get('error')
    message = request.args.get('message')

    if request.method == 'POST':
        is_json = request.is_json
        payload = request.get_json(silent=True) if is_json else request.form
        if not payload:
            payload = {}

        role = payload.get('role', 'student').strip().lower()
        if role not in ('student', 'admin'):
            role = 'student'

        if role == 'admin':
            identifier = payload.get('username') or payload.get('identifier') or payload.get('email', '')
            identifier = identifier.strip()
            password = payload.get('password', '').strip()
            if not identifier or not password:
                err = "Admin username/email and password are required."
                if is_json:
                    return jsonify({'status': False, 'message': err}), 400
                return render_template('login.html', error=err, initial_role='admin', initial_mode='login')

            if database.verify_admin(identifier, password):
                admin = database.get_admin_by_identifier(identifier)
                admin_name = admin['username'] if admin else identifier
                session.clear()
                session['admin_user'] = admin_name
                session['role'] = 'admin'
                session['auth_provider'] = 'local'
                session['user'] = {
                    'name': admin_name,
                    'role': 'admin',
                    'provider': 'local',
                    'email': admin.get('email') if admin else f"{admin_name}@visionattend.local"
                }
                if is_json:
                    return jsonify({'status': True, 'redirect': url_for('dashboard')})
                return redirect(url_for('dashboard'))
            else:
                err = "Invalid admin username or password. Please try again."
                if is_json:
                    return jsonify({'status': False, 'message': err}), 401
                return render_template('login.html', error=err, initial_role='admin', initial_mode='login')

        else:  # Student Login
            identifier = payload.get('student_identifier') or payload.get('identifier') or payload.get('roll_number') or payload.get('email', '')
            identifier = identifier.strip()
            password = payload.get('password', '').strip()
            if not identifier:
                err = "Please enter your Student Roll Number or Email."
                if is_json:
                    return jsonify({'status': False, 'message': err}), 400
                return render_template('login.html', error=err, initial_role='student', initial_mode='login')

            success, res = database.verify_student(identifier, password)
            if success:
                student = res
                session.clear()
                session['student_user'] = student['name']
                session['student_roll'] = student['roll_number']
                session['role'] = 'student'
                session['auth_provider'] = 'local'
                session['user'] = {
                    'name': student['name'],
                    'roll_number': student['roll_number'],
                    'department': student['department'],
                    'email': student['email'],
                    'role': 'student',
                    'provider': 'local'
                }
                if is_json:
                    return jsonify({'status': True, 'redirect': url_for('student_portal')})
                return redirect(url_for('student_portal'))
            else:
                err = res if isinstance(res, str) else "Invalid student credentials."
                if is_json:
                    return jsonify({'status': False, 'message': err}), 401
                return render_template('login.html', error=err, initial_role='student', initial_mode='login')

    return render_template('login.html', error=error, message=message, initial_role=initial_role, initial_mode=initial_mode)

@app.route('/login/student', methods=['POST'])
def student_login():
    """Authenticate student via Roll Number or Email (Form submission compatibility)."""
    identifier = request.form.get('student_identifier', '').strip()
    password = request.form.get('password', '').strip()
    if not identifier:
        return render_template('login.html', error="Please provide a Student Roll Number or Email.", initial_role='student', initial_mode='login')

    success, res = database.verify_student(identifier, password)
    if success:
        student = res
        session.clear()
        session['student_user'] = student['name']
        session['student_roll'] = student['roll_number']
        session['role'] = 'student'
        session['auth_provider'] = 'local'
        session['user'] = {
            'name': student['name'],
            'roll_number': student['roll_number'],
            'department': student['department'],
            'email': student['email'],
            'role': 'student',
            'provider': 'local'
        }
        return redirect(url_for('student_portal'))
    else:
        err = res if isinstance(res, str) else f"Student '{identifier}' not found in the database. Please register first."
        return render_template('login.html', error=err, initial_role='student', initial_mode='login')

@app.route('/register', methods=['POST'])
def register():
    """
    Handles registration for both Admin and Student accounts.
    Strictly displays 'Account already registered' if an account with the same
    Roll Number, Username, or Email already exists.
    """
    is_json = request.is_json
    payload = request.get_json(silent=True) if is_json else request.form
    if not payload:
        payload = {}

    role = payload.get('role', 'student').strip().lower()
    if role not in ('student', 'admin'):
        role = 'student'

    if role == 'admin':
        username = payload.get('username', '').strip()
        email = payload.get('email', '').strip()
        password = payload.get('password', '').strip()
        firebase_uid = payload.get('firebase_uid', '').strip()

        if not username or not password:
            err = "Admin username and password are required."
            if is_json:
                return jsonify({'status': False, 'message': err}), 400
            return render_template('login.html', error=err, initial_role='admin', initial_mode='register')

        success, msg = database.register_admin(username, password, email=email, firebase_uid=firebase_uid)
        if success:
            session.clear()
            session['admin_user'] = username
            session['role'] = 'admin'
            session['auth_provider'] = 'firebase' if firebase_uid else 'local'
            session['user'] = {
                'name': username,
                'role': 'admin',
                'provider': 'firebase' if firebase_uid else 'local',
                'email': email or f"{username}@visionattend.local"
            }
            if is_json:
                return jsonify({'status': True, 'redirect': url_for('dashboard')})
            return redirect(url_for('dashboard'))
        else:
            if is_json:
                return jsonify({'status': False, 'message': msg}), 400
            return render_template('login.html', error=msg, initial_role='admin', initial_mode='register')

    else:  # Student registration
        name = payload.get('name', '').strip()
        roll = payload.get('roll_number', '').strip().upper()
        department = payload.get('department', '').strip()
        email = payload.get('email', '').strip()
        password = payload.get('password', '').strip()
        firebase_uid = payload.get('firebase_uid', '').strip()

        if not name or not roll or not department:
            err = "Name, Roll Number, and Department are required."
            if is_json:
                return jsonify({'status': False, 'message': err}), 400
            return render_template('login.html', error=err, initial_role='student', initial_mode='register')

        success, msg = database.register_student_user(
            name=name,
            roll_number=roll,
            department=department,
            email=email,
            password=password,
            firebase_uid=firebase_uid
        )
        if success:
            session.clear()
            session['student_user'] = name
            session['student_roll'] = roll
            session['role'] = 'student'
            session['auth_provider'] = 'firebase' if firebase_uid else 'local'
            session['user'] = {
                'name': name,
                'roll_number': roll,
                'department': department,
                'email': email,
                'role': 'student',
                'provider': 'firebase' if firebase_uid else 'local'
            }
            if is_json:
                return jsonify({'status': True, 'redirect': url_for('student_portal')})
            return redirect(url_for('student_portal'))
        else:
            if is_json:
                return jsonify({'status': False, 'message': msg}), 400
            return render_template('login.html', error=msg, initial_role='student', initial_mode='register')

@app.route('/api/firebase_auth', methods=['POST'])
def api_firebase_auth():
    """
    JSON API for Firebase Authentication (Google popup or Firebase Email/Password).
    Synchronizes authenticated user identity with the local database.
    Strictly checks duplicate registration returning 'Account already registered'.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({'status': False, 'message': 'Invalid JSON request payload.'}), 400

    action = data.get('action', 'login').strip().lower()
    role = data.get('role', 'student').strip().lower()
    if role not in ('student', 'admin'):
        role = 'student'

    uid = data.get('uid', '').strip()
    email = data.get('email', '').strip()
    name = data.get('name', '').strip()
    roll_number = data.get('roll_number', '').strip().upper()
    department = data.get('department', '').strip()
    password = data.get('password', '').strip()

    if not uid and not email:
        return jsonify({'status': False, 'message': 'Firebase UID or email is required.'}), 400

    if action == 'register':
        if role == 'admin':
            existing = database.get_admin_by_uid(uid) if uid else None
            if not existing and email:
                existing = database.get_admin_by_identifier(email)
            if existing:
                return jsonify({'status': False, 'message': 'Account already registered'}), 400

            admin_user = name or (email.split('@')[0] if email else 'admin_user')
            success, msg = database.register_admin(admin_user, password or 'firebase_authenticated', email=email, firebase_uid=uid)
            if not success:
                return jsonify({'status': False, 'message': msg}), 400

            session.clear()
            session['admin_user'] = admin_user
            session['role'] = 'admin'
            session['auth_provider'] = 'firebase'
            session['user'] = {
                'name': admin_user,
                'email': email,
                'role': 'admin',
                'provider': 'firebase',
                'uid': uid
            }
            return jsonify({'status': True, 'redirect': url_for('dashboard')})

        else:  # Student Registration
            clean_roll = roll_number.strip().upper() if roll_number else ''
            clean_email = email.strip() if email else ''

            if clean_roll and database.get_student_by_roll(clean_roll):
                return jsonify({'status': False, 'message': 'Account already registered'}), 400
            if clean_email and database.get_student_by_email(clean_email):
                return jsonify({'status': False, 'message': 'Account already registered'}), 400
            if uid and database.get_student_by_uid(uid):
                return jsonify({'status': False, 'message': 'Account already registered'}), 400

            assigned_roll = clean_roll or f"STU-{abs(hash(clean_email or uid)) % 9000 + 1000}"
            dept = department or 'AI & Data Science'
            stu_name = name or (clean_email.split('@')[0].capitalize() if clean_email else 'Student')

            success, msg = database.register_student_user(
                name=stu_name,
                roll_number=assigned_roll,
                department=dept,
                email=clean_email,
                password=password,
                firebase_uid=uid
            )
            if not success:
                return jsonify({'status': False, 'message': msg}), 400

            session.clear()
            session['student_user'] = stu_name
            session['student_roll'] = assigned_roll
            session['role'] = 'student'
            session['auth_provider'] = 'firebase'
            session['user'] = {
                'name': stu_name,
                'roll_number': assigned_roll,
                'department': dept,
                'email': clean_email,
                'role': 'student',
                'provider': 'firebase',
                'uid': uid
            }
            return jsonify({'status': True, 'redirect': url_for('student_portal')})

    else:  # action == 'login'
        if role == 'admin':
            admin = database.get_admin_by_uid(uid) if uid else None
            if not admin and email:
                admin = database.get_admin_by_identifier(email)
            if admin:
                if not admin.get('firebase_uid') and uid:
                    database.link_admin_firebase(admin['username'], uid)
                session.clear()
                session['admin_user'] = admin['username']
                session['role'] = 'admin'
                session['auth_provider'] = 'firebase'
                session['user'] = {
                    'name': admin['username'],
                    'role': 'admin',
                    'email': admin.get('email', email),
                    'provider': 'firebase',
                    'uid': uid
                }
                return jsonify({'status': True, 'redirect': url_for('dashboard')})
            else:
                return jsonify({'status': False, 'message': 'Admin account not found. Please register first.'}), 404

        else:  # student login
            student = database.get_student_by_uid(uid) if uid else None
            if not student and email:
                student = database.get_student_by_email(email)
            if not student and roll_number:
                student = database.get_student_by_roll(roll_number)
            if student:
                if not student.get('firebase_uid') and uid:
                    database.link_student_firebase(student['roll_number'], uid)
                session.clear()
                session['student_user'] = student['name']
                session['student_roll'] = student['roll_number']
                session['role'] = 'student'
                session['auth_provider'] = 'firebase'
                session['user'] = {
                    'name': student['name'],
                    'roll_number': student['roll_number'],
                    'department': student['department'],
                    'email': student.get('email', email),
                    'role': 'student',
                    'provider': 'firebase',
                    'uid': uid
                }
                return jsonify({'status': True, 'redirect': url_for('student_portal')})
            else:
                return jsonify({'status': False, 'message': 'Student account not found in database. Please register first.'}), 404

# ----------------------------------------------------
# Social OAuth Routes (Google & GitHub)
# ----------------------------------------------------
def finish_oauth_login(provider, provider_id, email, name, role, avatar=''):
    """Shared authentication finalize helper for social logins."""
    role = role.lower() if role in ('student', 'admin') else 'student'
    session.clear()

    if role == 'admin':
        session['admin_user'] = name
        session['role'] = 'admin'
        session['auth_provider'] = provider
        session['user'] = {
            'name': name,
            'email': email,
            'role': 'admin',
            'provider': provider,
            'avatar': avatar
        }
        database.record_oauth_login(provider, provider_id, email, name, 'admin', avatar_url=avatar)
        return redirect(url_for('dashboard'))
    else:
        student = database.get_student_by_email(email)
        if not student:
            # Check if default demo student can be claimed, else generate standard ID
            demo_student = database.get_student_by_roll('CS-2024-001')
            if demo_student and (not demo_student.get('email') or demo_student.get('email') == email):
                roll_number = 'CS-2024-001'
            else:
                roll_number = f"STU-{abs(hash(email)) % 9000 + 1000}"
                database.add_student(roll_number, name, 'Computer Science & AI', email=email)
            student = database.get_student_by_roll(roll_number) or {'roll_number': roll_number, 'name': name, 'department': 'Computer Science & AI'}

        roll_number = student.get('roll_number', 'CS-2024-001')
        session['student_user'] = name
        session['student_roll'] = roll_number
        session['role'] = 'student'
        session['auth_provider'] = provider
        session['user'] = {
            'name': name,
            'email': email,
            'roll_number': roll_number,
            'department': student.get('department', 'Computer Science & AI'),
            'role': 'student',
            'provider': provider,
            'avatar': avatar
        }
        database.record_oauth_login(provider, provider_id, email, name, 'student', roll_number=roll_number, avatar_url=avatar)
        return redirect(url_for('student_portal'))

@app.route('/auth/google')
def auth_google():
    """Initiates Google OAuth 2.0 flow or fallback simulator."""
    role = request.args.get('role', 'student').lower()
    if role not in ('student', 'admin'):
        role = 'student'

    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        redirect_uri = url_for('auth_google_callback', _external=True)
        state = f"{role}:{os.urandom(8).hex()}"
        session['oauth_state'] = state
        params = {
            'client_id': GOOGLE_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': 'openid email profile',
            'state': state,
            'prompt': 'select_account'
        }
        return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}")
    else:
        # Launch instant simulator modal
        return redirect(url_for('login', simulate='1', provider='google', role=role))

@app.route('/auth/google/callback')
def auth_google_callback():
    """Handles Google OAuth authorization callback."""
    error = request.args.get('error')
    if error:
        return redirect(url_for('login', error=f"Google authentication cancelled: {error}"))

    code = request.args.get('code')
    state = request.args.get('state', '')
    role = state.split(':')[0] if ':' in state else 'student'

    if not code:
        return redirect(url_for('login', error="No authorization code received from Google."))

    redirect_uri = url_for('auth_google_callback', _external=True)
    token_resp = requests.post(
        'https://oauth2.googleapis.com/token',
        data={
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri
        },
        timeout=10
    )
    if token_resp.status_code != 200:
        return redirect(url_for('login', error="Failed to exchange authorization token with Google."))

    tokens = token_resp.json()
    access_token = tokens.get('access_token')
    user_resp = requests.get(
        'https://www.googleapis.com/oauth2/v2/userinfo',
        headers={'Authorization': f"Bearer {access_token}"},
        timeout=10
    )
    if user_resp.status_code != 200:
        return redirect(url_for('login', error="Failed to retrieve Google profile data."))

    user_info = user_resp.json()
    email = user_info.get('email', '')
    name = user_info.get('name') or email.split('@')[0].capitalize()
    google_id = user_info.get('id', '')
    picture = user_info.get('picture', '')

    return finish_oauth_login(provider='google', provider_id=google_id, email=email, name=name, role=role, avatar=picture)

@app.route('/auth/github')
def auth_github():
    """Initiates GitHub OAuth 2.0 flow or fallback simulator."""
    role = request.args.get('role', 'student').lower()
    if role not in ('student', 'admin'):
        role = 'student'

    if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
        redirect_uri = url_for('auth_github_callback', _external=True)
        state = f"{role}:{os.urandom(8).hex()}"
        session['oauth_state'] = state
        params = {
            'client_id': GITHUB_CLIENT_ID,
            'redirect_uri': redirect_uri,
            'scope': 'read:user user:email',
            'state': state
        }
        return redirect(f"https://github.com/login/oauth/authorize?{urllib.parse.urlencode(params)}")
    else:
        return redirect(url_for('login', simulate='1', provider='github', role=role))

@app.route('/auth/github/callback')
def auth_github_callback():
    """Handles GitHub OAuth authorization callback."""
    error = request.args.get('error')
    if error:
        return redirect(url_for('login', error=f"GitHub authentication cancelled: {error}"))

    code = request.args.get('code')
    state = request.args.get('state', '')
    role = state.split(':')[0] if ':' in state else 'student'

    if not code:
        return redirect(url_for('login', error="No authorization code received from GitHub."))

    redirect_uri = url_for('auth_github_callback', _external=True)
    token_resp = requests.post(
        'https://github.com/login/oauth/access_token',
        headers={'Accept': 'application/json'},
        data={
            'client_id': GITHUB_CLIENT_ID,
            'client_secret': GITHUB_CLIENT_SECRET,
            'code': code,
            'redirect_uri': redirect_uri
        },
        timeout=10
    )
    if token_resp.status_code != 200:
        return redirect(url_for('login', error="Failed to exchange authorization token with GitHub."))

    tokens = token_resp.json()
    access_token = tokens.get('access_token')
    if not access_token:
        return redirect(url_for('login', error="GitHub did not grant an access token."))

    user_resp = requests.get(
        'https://api.github.com/user',
        headers={'Authorization': f"Bearer {access_token}", 'Accept': 'application/json'},
        timeout=10
    )
    if user_resp.status_code != 200:
        return redirect(url_for('login', error="Failed to fetch GitHub profile."))

    github_user = user_resp.json()
    email = github_user.get('email')
    if not email:
        emails_resp = requests.get(
            'https://api.github.com/user/emails',
            headers={'Authorization': f"Bearer {access_token}", 'Accept': 'application/json'},
            timeout=10
        )
        if emails_resp.status_code == 200:
            for em in emails_resp.json():
                if em.get('primary'):
                    email = em.get('email')
                    break
    email = email or f"{github_user.get('login')}@users.noreply.github.com"
    name = github_user.get('name') or github_user.get('login') or 'GitHub User'
    github_id = str(github_user.get('id', ''))
    avatar = github_user.get('avatar_url', '')

    return finish_oauth_login(provider='github', provider_id=github_id, email=email, name=name, role=role, avatar=avatar)

@app.route('/auth/mock')
def mock_auth():
    """Instant 1-Click demo authentication for student or admin."""
    role = request.args.get('role', 'student').lower()
    provider = request.args.get('provider', 'google').lower()

    if role == 'admin':
        return finish_oauth_login(
            provider=provider,
            provider_id=f"demo_admin_{provider}",
            email=f"admin.vision@{provider}.com",
            name="System Administrator",
            role="admin",
            avatar="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=256&q=80"
        )
    else:
        student = database.get_student_by_roll('CS-2024-001')
        name = student['name'] if student else "Alex Johnson"
        email = student.get('email') if student else f"alex.student@{provider}.com"
        return finish_oauth_login(
            provider=provider,
            provider_id=f"demo_student_{provider}",
            email=email,
            name=name,
            role="student",
            avatar="https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=256&q=80"
        )

@app.route('/auth/mock_post', methods=['POST'])
def mock_auth_post():
    """Processes interactive modal submission for simulated social login."""
    provider = request.form.get('provider', 'google').lower()
    role = request.form.get('role', 'student').lower()
    name = request.form.get('name', '').strip() or ('Alex Johnson' if role == 'student' else 'System Administrator')
    email = request.form.get('email', '').strip() or f"user@{provider}.com"
    roll_number = request.form.get('roll_number', '').strip().upper()

    if role == 'student' and roll_number:
        if not database.get_student_by_roll(roll_number):
            database.add_student(roll_number, name, 'Computer Science & AI', email=email)

    return finish_oauth_login(
        provider=provider,
        provider_id=f"sim_{abs(hash(email))}",
        email=email,
        name=name,
        role=role
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', message="You have been signed out successfully."))

# ----------------------------------------------------
# Student Portal Routes
# ----------------------------------------------------
@app.route('/student/portal')
def student_portal():
    """Personalized attendance portal for students."""
    if session.get('role') != 'student':
        if session.get('admin_user'):
            roll = 'CS-2024-001'
        else:
            return redirect(url_for('login', role='student', error="Please sign in to access your student attendance portal."))
    else:
        roll = session.get('student_roll', 'CS-2024-001')

    summary = database.get_student_summary(roll)
    if not summary:
        all_students = database.get_all_students()
        if all_students:
            roll = all_students[0]['roll_number']
            session['student_roll'] = roll
            summary = database.get_student_summary(roll)

    records = database.get_student_attendance(roll)
    today_str = datetime.now().strftime('%A, %B %d, %Y')

    student_info = summary['student'] if summary else {
        'name': session.get('student_user', 'Student'),
        'roll_number': roll,
        'department': 'Computer Science & AI',
        'email': session.get('user', {}).get('email', ''),
        'embedding': None,
        'photo_path': None
    }

    return render_template(
        'student_portal.html',
        active_page='student_portal',
        student=student_info,
        summary=summary or {
            'attendance_rate': 0.0,
            'present_count': 0,
            'absent_count': 0,
            'total_classes': 0,
            'is_present_today': False,
            'today_record': None
        },
        attendance_records=records,
        today_date=today_str,
        user_session=session.get('user', {})
    )

@app.route('/student/export_csv')
def student_export_csv():
    """Exports personal student attendance to CSV."""
    roll = session.get('student_roll', 'CS-2024-001')
    records = database.get_student_attendance(roll)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'Roll Number', 'Student Name', 'Department', 'Date', 'Time', 'Status', 'Confidence (%)'])
    for r in records:
        writer.writerow([r['id'], r['roll_number'], r['name'], r['department'], r['date'], r['time'], r['status'], r['confidence']])

    filename = f"My_Attendance_{roll}_{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@app.route('/student/export_excel')
def student_export_excel():
    """Exports personal student attendance to styled Excel (.xlsx)."""
    roll = session.get('student_roll', 'CS-2024-001')
    records = database.get_student_attendance(roll)

    df = pd.DataFrame(records)
    if not df.empty:
        rename_map = {
            'id': 'Record ID',
            'roll_number': 'Roll Number',
            'name': 'Student Name',
            'department': 'Department',
            'date': 'Attendance Date',
            'time': 'Time Logged',
            'status': 'Status',
            'confidence': 'AI Match Confidence (%)'
        }
        df = df[[col for col in rename_map.keys() if col in df.columns]]
        df = df.rename(columns=rename_map)
    else:
        df = pd.DataFrame(columns=['Record ID', 'Roll Number', 'Student Name', 'Department', 'Attendance Date', 'Time Logged', 'Status', 'AI Match Confidence (%)'])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='My Attendance', index=False)
    excel_buffer.seek(0)

    filename = f"My_Attendance_{roll}_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

# ----------------------------------------------------
# Pages (Role Access Managed)
# ----------------------------------------------------
@app.route('/')
def dashboard():
    if session.get('role') == 'student':
        return redirect(url_for('student_portal'))

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
    if session.get('role') == 'student':
        return redirect(url_for('student_portal'))

    student_list = database.get_all_students()
    return render_template(
        'students.html',
        active_page='students',
        students=student_list,
        admin_user=session.get('admin_user', 'Admin')
    )

@app.route('/reports')
def reports():
    if session.get('role') == 'student':
        return redirect(url_for('student_portal'))

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
@admin_required
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
@admin_required
def api_delete_student(roll_number):
    """
    Permanently deletes a student record. Strictly restricted to Administrators.
    """
    try:
        roll = roll_number.strip().upper()
        deleted = database.delete_student(roll)
        if deleted:
            photo_path = os.path.join(DATASET_DIR, f"{roll}.jpg")
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                except Exception:
                    pass
            
            attendance_system.reload_enrolled_students()
            return jsonify({'status': True, 'message': f'Student {roll} deleted successfully.'})
        else:
            return jsonify({'status': False, 'message': f'Student {roll} not found.'}), 404
    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500

@app.route('/api/update_student/<roll_number>', methods=['PUT', 'POST'])
@admin_required
def api_update_student(roll_number):
    """
    Updates student information. Strictly restricted to Administrators.
    Accepts JSON payload: name, roll_number (new), department, email, optional image.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'status': False, 'message': 'Invalid JSON request payload.'}), 400

        orig_roll = roll_number.strip().upper()
        name = data.get('name', '').strip()
        new_roll = data.get('roll_number', '').strip().upper() or orig_roll
        department = data.get('department', '').strip()
        email = data.get('email', '').strip()
        image_data = data.get('image', '')

        if not name or not new_roll or not department:
            return jsonify({'status': False, 'message': 'Name, Roll Number, and Department are required.'}), 400

        new_embedding = None
        photo_path = None

        if image_data:
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            raw_dec = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
            if raw_dec is not None:
                engine = attendance_system.face_engine
                faces = engine.detect_faces(raw_dec)
                if faces:
                    target_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                    emb = engine.extract_feature(raw_dec, target_face['raw_face'])
                    if emb is not None:
                        new_embedding = emb.astype(np.float32).tobytes()
                        photo_filename = f"{new_roll}.jpg"
                        photo_path = os.path.join(DATASET_DIR, photo_filename)
                        cv2.imwrite(photo_path, raw_dec)

        # If roll number changed without new photo, rename existing dataset photo if present
        if new_roll != orig_roll and not photo_path:
            old_photo = os.path.join(DATASET_DIR, f"{orig_roll}.jpg")
            new_photo = os.path.join(DATASET_DIR, f"{new_roll}.jpg")
            if os.path.exists(old_photo):
                try:
                    os.replace(old_photo, new_photo)
                    photo_path = new_photo
                except Exception:
                    pass

        success, msg = database.update_student(
            original_roll=orig_roll,
            name=name,
            new_roll=new_roll,
            department=department,
            email=email,
            embedding=new_embedding,
            photo_path=photo_path
        )

        if success:
            attendance_system.reload_enrolled_students()
            return jsonify({'status': True, 'message': msg})
        else:
            return jsonify({'status': False, 'message': msg}), 400

    except Exception as e:
        return jsonify({'status': False, 'message': f'Update error: {str(e)}'}), 500

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
    port = int(os.environ.get('PORT', '5000'))
    print("=" * 65)
    print(" VisionAttend - AI Real-Time Facial Attendance System")
    print(f" Dashboard active on: http://0.0.0.0:{port}")
    print(" Set ADMIN_USERNAME and ADMIN_PASSWORD before first launch.")
    print("=" * 65)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
