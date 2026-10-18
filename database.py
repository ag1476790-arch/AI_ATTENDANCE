import sqlite3
import os
import hmac
from datetime import datetime
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'attendance.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON;')
    conn.execute('PRAGMA journal_mode = WAL;')
    conn.execute('PRAGMA synchronous = NORMAL;')
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Students Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            email TEXT DEFAULT '',
            embedding BLOB,
            photo_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance Table (Unique constraint prevents duplicate entries for the same student on the same day)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            roll_number TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            confidence REAL DEFAULT 0.0,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            UNIQUE(roll_number, date)
        )
    ''')

    # Admin Table for Authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # OAuth Accounts Table for Social Logins (Google & GitHub)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS oauth_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            email TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            roll_number TEXT DEFAULT '',
            avatar_url TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider, provider_id, role)
        )
    ''')

    # Dynamic schema migrations for students (password and firebase_uid)
    cursor.execute("PRAGMA table_info(students)")
    student_cols = [row['name'] for row in cursor.fetchall()]
    if 'password' not in student_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN password TEXT DEFAULT ''")
    if 'firebase_uid' not in student_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN firebase_uid TEXT DEFAULT ''")

    # Dynamic schema migrations for admins (email and firebase_uid)
    cursor.execute("PRAGMA table_info(admins)")
    admin_cols = [row['name'] for row in cursor.fetchall()]
    if 'email' not in admin_cols:
        cursor.execute("ALTER TABLE admins ADD COLUMN email TEXT DEFAULT ''")
    if 'firebase_uid' not in admin_cols:
        cursor.execute("ALTER TABLE admins ADD COLUMN firebase_uid TEXT DEFAULT ''")

    # Create the first admin if none exists
    cursor.execute('SELECT COUNT(*) as count FROM admins')
    if cursor.fetchone()['count'] == 0:
        username = os.environ.get('ADMIN_USERNAME', 'admin').strip()
        password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        cursor.execute(
            'INSERT INTO admins (username, password, email) VALUES (?, ?, ?)',
            (username, generate_password_hash(password), 'admin@visionattend.local')
        )

    conn.commit()
    conn.close()

def verify_admin(identifier, password):
    """
    Verifies administrator credentials by Username or Email.
    Returns True if valid, False otherwise.
    """
    if not identifier or not password:
        return False
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean_id = identifier.strip()
        cursor.execute(
            '''SELECT id, password FROM admins 
               WHERE LOWER(username) = LOWER(?) OR (email != '' AND LOWER(email) = LOWER(?))''',
            (clean_id, clean_id)
        )
        admin = cursor.fetchone()
        if admin is None:
            return False

        stored_password = admin['password']
        submitted_password = password.strip()
        if stored_password.startswith(('scrypt:', 'pbkdf2:')):
            return check_password_hash(stored_password, submitted_password)

        if hmac.compare_digest(stored_password, submitted_password):
            cursor.execute(
                'UPDATE admins SET password = ? WHERE id = ?',
                (generate_password_hash(submitted_password), admin['id'])
            )
            conn.commit()
            return True
        return False
    finally:
        conn.close()

def verify_student(identifier, password=''):
    """
    Verifies student credentials by Roll Number or Email.
    Returns: (success: bool, student_or_msg: dict/str)
    """
    if not identifier:
        return False, "Student Roll Number or Email is required."
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean_id = identifier.strip()
        cursor.execute('''
            SELECT * FROM students 
            WHERE LOWER(roll_number) = LOWER(?) OR (email != '' AND LOWER(email) = LOWER(?))
        ''', (clean_id, clean_id))
        student = cursor.fetchone()
        if not student:
            return False, f"Student account '{identifier}' not found in the database. Please register first."
        
        student_dict = dict(student)
        stored_pw = student_dict.get('password', '')
        
        # If student has a password set, verify it
        if stored_pw:
            if not password:
                return False, "Password is required to sign in."
            submitted = password.strip()
            if stored_pw.startswith(('scrypt:', 'pbkdf2:')):
                if check_password_hash(stored_pw, submitted):
                    return True, student_dict
                return False, "Incorrect password. Please try again."
            elif hmac.compare_digest(stored_pw, submitted):
                cursor.execute('UPDATE students SET password = ? WHERE id = ?', (generate_password_hash(submitted), student_dict['id']))
                conn.commit()
                return True, student_dict
            return False, "Incorrect password. Please try again."
        else:
            # If no password set yet (e.g. registered without password), set password if submitted
            if password:
                cursor.execute('UPDATE students SET password = ? WHERE id = ?', (generate_password_hash(password.strip()), student_dict['id']))
                conn.commit()
            return True, student_dict
    finally:
        conn.close()

def register_student_user(name, roll_number, department, email='', password='', firebase_uid=''):
    """
    Registers a new student user. If roll number or email is already taken,
    strictly returns (False, "Account already registered").
    """
    clean_name = name.strip()
    clean_roll = roll_number.strip().upper()
    clean_dept = department.strip()
    clean_email = email.strip()
    clean_pw = password.strip()

    if not clean_name or not clean_roll or not clean_dept:
        return False, "Name, Roll Number, and Department are required."

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Check duplicate by roll number
        cursor.execute('SELECT id FROM students WHERE LOWER(roll_number) = LOWER(?)', (clean_roll,))
        if cursor.fetchone():
            return False, "Account already registered"

        # Check duplicate by email if provided
        if clean_email:
            cursor.execute('SELECT id FROM students WHERE email != "" AND LOWER(email) = LOWER(?)', (clean_email,))
            if cursor.fetchone():
                return False, "Account already registered"

        # Check duplicate by firebase_uid if provided
        if firebase_uid:
            cursor.execute('SELECT id FROM students WHERE firebase_uid != "" AND firebase_uid = ?', (firebase_uid.strip(),))
            if cursor.fetchone():
                return False, "Account already registered"

        pw_hash = generate_password_hash(clean_pw) if clean_pw else ''
        cursor.execute('''
            INSERT INTO students (roll_number, name, department, email, password, firebase_uid)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (clean_roll, clean_name, clean_dept, clean_email, pw_hash, firebase_uid.strip()))
        conn.commit()
        return True, "Student registered successfully."
    except sqlite3.IntegrityError:
        return False, "Account already registered"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def has_admin_accounts():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM admins LIMIT 1')
        return cursor.fetchone() is not None
    finally:
        conn.close()

def add_student(roll_number, name, department, email='', embedding=None, photo_path=''):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO students (roll_number, name, department, email, embedding, photo_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (roll_number.strip().upper(), name.strip(), department.strip(), email.strip(), embedding, photo_path))
        conn.commit()
        return True, "Student registered successfully."
    except sqlite3.IntegrityError:
        return False, "Account already registered"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def update_student_embedding(roll_number, embedding, photo_path=''):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if photo_path:
            cursor.execute('''
                UPDATE students SET embedding = ?, photo_path = ? WHERE roll_number = ?
            ''', (embedding, photo_path, roll_number.strip().upper()))
        else:
            cursor.execute('''
                UPDATE students SET embedding = ? WHERE roll_number = ?
            ''', (embedding, roll_number.strip().upper()))
        conn.commit()
        return True
    finally:
        conn.close()

def get_all_students():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, roll_number, name, department, email, photo_path, created_at FROM students ORDER BY roll_number')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_student_by_roll(roll_number):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE roll_number = ?', (roll_number.strip().upper(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_embeddings():
    """Retrieve all student face embeddings for recognition matcher."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT roll_number, name, embedding FROM students WHERE embedding IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()
    return [(row['roll_number'], row['name'], row['embedding']) for row in rows]

def delete_student(roll_number):
    """
    Permanently deletes a student, their biometric records, 
    linked attendance history, and OAuth account records.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        roll = roll_number.strip().upper()
        # Delete related attendance records
        cursor.execute('DELETE FROM attendance WHERE roll_number = ?', (roll,))
        # Delete related oauth records
        cursor.execute('DELETE FROM oauth_accounts WHERE roll_number = ?', (roll,))
        # Delete student record
        cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        print(f"[DB Delete Error] {e}")
        return False
    finally:
        conn.close()

def update_student(original_roll, name=None, new_roll=None, department=None, email=None, embedding=None, photo_path=None):
    """
    Updates an enrolled student's details (Name, Roll, Department, Email, Biometrics).
    Keeps attendance records and OAuth links synchronized if Roll Number or Name changes.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        orig = original_roll.strip().upper()
        new_r = new_roll.strip().upper() if new_roll else orig

        cursor.execute('SELECT id, name, department, email, photo_path FROM students WHERE roll_number = ?', (orig,))
        curr = cursor.fetchone()
        if not curr:
            return False, f"Student with Roll Number '{orig}' does not exist."

        # Verify roll number uniqueness if changing roll
        if new_r != orig:
            cursor.execute('SELECT id FROM students WHERE roll_number = ?', (new_r,))
            if cursor.fetchone():
                return False, f"Roll Number '{new_r}' is already assigned to another student."

        updated_name = name.strip() if name else curr['name']
        updated_dept = department.strip() if department else curr['department']
        updated_email = email.strip() if email is not None else curr['email']
        updated_photo = photo_path if photo_path else curr['photo_path']

        # Update student record
        if embedding is not None:
            cursor.execute('''
                UPDATE students
                SET roll_number = ?, name = ?, department = ?, email = ?, embedding = ?, photo_path = ?
                WHERE roll_number = ?
            ''', (new_r, updated_name, updated_dept, updated_email, embedding, updated_photo, orig))
        else:
            cursor.execute('''
                UPDATE students
                SET roll_number = ?, name = ?, department = ?, email = ?
                WHERE roll_number = ?
            ''', (new_r, updated_name, updated_dept, updated_email, orig))

        # Cascade updates to attendance history table
        cursor.execute('''
            UPDATE attendance
            SET roll_number = ?, name = ?, department = ?
            WHERE roll_number = ?
        ''', (new_r, updated_name, updated_dept, orig))

        # Cascade updates to oauth_accounts table
        if new_r != orig:
            cursor.execute('''
                UPDATE oauth_accounts
                SET roll_number = ?
                WHERE roll_number = ?
            ''', (new_r, orig))

        conn.commit()
        return True, f"Student '{updated_name}' ({new_r}) updated successfully."
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def register_admin(username, password, email=None, firebase_uid=''):
    """Creates a new administrator account with salted PBKDF2/scrypt password hash."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        user = username.strip()
        pw = password.strip()
        em = email.strip() if email else ''
        if not user or not pw:
            return False, "Username and password are required."

        # Check duplicate username
        cursor.execute('SELECT id FROM admins WHERE LOWER(username) = LOWER(?)', (user,))
        if cursor.fetchone():
            return False, "Account already registered"

        # Check duplicate email if provided
        if em:
            cursor.execute('SELECT id FROM admins WHERE email != "" AND LOWER(email) = LOWER(?)', (em,))
            if cursor.fetchone():
                return False, "Account already registered"

        # Check duplicate firebase_uid if provided
        if firebase_uid:
            cursor.execute('SELECT id FROM admins WHERE firebase_uid != "" AND firebase_uid = ?', (firebase_uid.strip(),))
            if cursor.fetchone():
                return False, "Account already registered"

        cursor.execute(
            'INSERT INTO admins (username, password, email, firebase_uid) VALUES (?, ?, ?, ?)',
            (user, generate_password_hash(pw), em, firebase_uid.strip())
        )
        conn.commit()
        return True, f"Admin account '{user}' registered successfully."
    except sqlite3.IntegrityError:
        return False, "Account already registered"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def get_admin_by_identifier(identifier):
    """Retrieves admin record by username or email."""
    if not identifier:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean = identifier.strip().lower()
        cursor.execute(
            'SELECT id, username, email, firebase_uid, created_at FROM admins WHERE LOWER(username) = ? OR (email != "" AND LOWER(email) = ?)',
            (clean, clean)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_admin_by_uid(firebase_uid):
    """Retrieves admin record by Firebase UID."""
    if not firebase_uid:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'SELECT id, username, email, firebase_uid, created_at FROM admins WHERE firebase_uid != "" AND firebase_uid = ?',
            (firebase_uid.strip(),)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_student_by_uid(firebase_uid):
    """Retrieves student record by Firebase UID."""
    if not firebase_uid:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'SELECT * FROM students WHERE firebase_uid != "" AND firebase_uid = ?',
            (firebase_uid.strip(),)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def link_admin_firebase(identifier, firebase_uid):
    """Links a Firebase UID to an existing admin account."""
    if not identifier or not firebase_uid:
        return False
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean = identifier.strip().lower()
        cursor.execute(
            'UPDATE admins SET firebase_uid = ? WHERE (LOWER(username) = ? OR LOWER(email) = ?) AND (firebase_uid IS NULL OR firebase_uid = "")',
            (firebase_uid.strip(), clean, clean)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def link_student_firebase(identifier, firebase_uid):
    """Links a Firebase UID to an existing student record."""
    if not identifier or not firebase_uid:
        return False
    conn = get_connection()
    cursor = conn.cursor()
    try:
        clean = identifier.strip().lower()
        cursor.execute(
            'UPDATE students SET firebase_uid = ? WHERE (LOWER(roll_number) = ? OR LOWER(email) = ?) AND (firebase_uid IS NULL OR firebase_uid = "")',
            (firebase_uid.strip(), clean, clean)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def mark_attendance(roll_number, confidence=0.0):
    """
    Attempts to mark attendance for today.
    Returns: (success: bool, message: str, student_info: dict)
    """
    student = get_student_by_roll(roll_number)
    if not student:
        return False, f"Student '{roll_number}' not found in database.", None

    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%I:%M:%S %p')

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO attendance (student_id, roll_number, name, department, date, time, status, confidence)
            VALUES (?, ?, ?, ?, ?, ?, 'Present', ?)
        ''', (student['id'], student['roll_number'], student['name'], student['department'], today_str, time_str, round(float(confidence), 2)))
        conn.commit()
        return True, f"Attendance marked for {student['name']} ({student['roll_number']}) at {time_str}.", {
            'roll_number': student['roll_number'],
            'name': student['name'],
            'department': student['department'],
            'date': today_str,
            'time': time_str,
            'status': 'Present',
            'confidence': round(float(confidence), 2)
        }
    except sqlite3.IntegrityError:
        # Already marked today
        return False, f"Attendance already marked today for {student['name']}.", {
            'roll_number': student['roll_number'],
            'name': student['name'],
            'department': student['department'],
            'date': today_str,
            'status': 'Already Marked'
        }
    except Exception as e:
        return False, f"Database error: {str(e)}", None
    finally:
        conn.close()

def get_today_attendance():
    today_str = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.id, a.roll_number, a.name, a.department, a.date, a.time, a.status, a.confidence, s.photo_path
        FROM attendance a
        LEFT JOIN students s ON a.roll_number = s.roll_number
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (today_str,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_attendance_reports(start_date=None, end_date=None, roll_number=None, department=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = '''
        SELECT a.id, a.roll_number, a.name, a.department, a.date, a.time, a.status, a.confidence
        FROM attendance a
        WHERE 1=1
    '''
    params = []
    if start_date:
        query += ' AND a.date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND a.date <= ?'
        params.append(end_date)
    if roll_number:
        query += ' AND a.roll_number = ?'
        params.append(roll_number.strip().upper())
    if department:
        query += ' AND a.department = ?'
        params.append(department.strip())
    
    query += ' ORDER BY a.date DESC, a.time DESC'
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_dashboard_stats():
    today_str = datetime.now().strftime('%Y-%m-%d')
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM students')
    total_students = cursor.fetchone()['count']
    
    cursor.execute('SELECT COUNT(*) as count FROM attendance WHERE date = ?', (today_str,))
    present_today = cursor.fetchone()['count']
    
    absent_today = max(0, total_students - present_today)
    attendance_rate = round((present_today / total_students * 100), 1) if total_students > 0 else 0.0
    
    conn.close()
    return {
        'total_students': total_students,
        'present_today': present_today,
        'absent_today': absent_today,
        'attendance_rate': attendance_rate,
        'today_date': today_str
    }

def get_student_analytics():
    """Calculates attendance percentages for each registered student."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total unique class dates on record
    cursor.execute('SELECT COUNT(DISTINCT date) as count FROM attendance')
    total_dates = cursor.fetchone()['count'] or 1
    
    cursor.execute('''
        SELECT s.roll_number, s.name, s.department, s.photo_path,
               COUNT(a.id) as days_present
        FROM students s
        LEFT JOIN attendance a ON s.roll_number = a.roll_number
        GROUP BY s.id
        ORDER BY s.roll_number
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    analytics = []
    for row in rows:
        days = row['days_present']
        pct = round((days / total_dates * 100), 1)
        analytics.append({
            'roll_number': row['roll_number'],
            'name': row['name'],
            'department': row['department'],
            'photo_path': row['photo_path'],
            'days_present': days,
            'total_days': total_dates,
            'percentage': pct
        })
    return analytics

def get_student_by_email(email):
    """Retrieve student record by email address."""
    if not email:
        return None
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students WHERE LOWER(email) = LOWER(?)', (email.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_student_attendance(roll_number):
    """Retrieve attendance logs for a specific student."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, roll_number, name, department, date, time, status, confidence
        FROM attendance
        WHERE roll_number = ?
        ORDER BY date DESC, time DESC
    ''', (roll_number.strip().upper(),))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_student_summary(roll_number):
    """Calculate personalized attendance metrics for the student portal."""
    roll = roll_number.strip().upper()
    student = get_student_by_roll(roll)
    if not student:
        return None
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total unique class sessions recorded across system
    cursor.execute('SELECT COUNT(DISTINCT date) as count FROM attendance')
    total_dates = cursor.fetchone()['count'] or 0
    
    # Classes attended by this student
    cursor.execute('SELECT COUNT(*) as count FROM attendance WHERE roll_number = ?', (roll,))
    attended = cursor.fetchone()['count'] or 0
    
    # Today's check-in record
    today_str = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('SELECT * FROM attendance WHERE roll_number = ? AND date = ?', (roll, today_str))
    today_record = cursor.fetchone()
    
    conn.close()
    
    effective_total = max(total_dates, attended, 1)
    pct = round((attended / effective_total * 100), 1)
    absent = max(0, effective_total - attended)
    
    return {
        'student': student,
        'total_classes': effective_total,
        'present_count': attended,
        'absent_count': absent,
        'attendance_rate': pct,
        'today_record': dict(today_record) if today_record else None,
        'is_present_today': today_record is not None
    }

def record_oauth_login(provider, provider_id, email, name, role, roll_number='', avatar_url=''):
    """Records or updates social OAuth login session in database."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute('''
            INSERT INTO oauth_accounts (provider, provider_id, email, name, role, roll_number, avatar_url, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, provider_id, role) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                avatar_url = excluded.avatar_url,
                roll_number = CASE WHEN excluded.roll_number != '' THEN excluded.roll_number ELSE oauth_accounts.roll_number END,
                last_login = excluded.last_login
        ''', (provider, str(provider_id), email.strip(), name.strip(), role.strip().lower(), roll_number.strip().upper(), avatar_url.strip(), now_str))
        conn.commit()
    except Exception as e:
        print(f"[OAuth DB Error] {e}")
    finally:
        conn.close()

def ensure_sample_student_if_empty():
    """Seeds a sample demo student if database has no students."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as count FROM students')
    count = cursor.fetchone()['count']
    if count == 0:
        cursor.execute('''
            INSERT INTO students (roll_number, name, department, email)
            VALUES (?, ?, ?, ?)
        ''', ('CS-2024-001', 'Alex Johnson', 'Computer Science & AI', 'alex.student@gmail.com'))
        student_id = cursor.lastrowid
        
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%I:%M:%S %p')
        cursor.execute('''
            INSERT OR IGNORE INTO attendance (student_id, roll_number, name, department, date, time, status, confidence)
            VALUES (?, ?, ?, ?, ?, ?, 'Present', 97.4)
        ''', (student_id, 'CS-2024-001', 'Alex Johnson', 'Computer Science & AI', today_str, time_str))
        conn.commit()
    conn.close()

# Initialize on import
init_db()
ensure_sample_student_if_empty()
