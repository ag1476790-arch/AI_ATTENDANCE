import sqlite3
import os
import hmac
from datetime import datetime
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'attendance.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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

    # Create the first admin if none exists
    cursor.execute('SELECT COUNT(*) as count FROM admins')
    if cursor.fetchone()['count'] == 0:
        username = os.environ.get('ADMIN_USERNAME', 'admin').strip()
        password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        cursor.execute(
            'INSERT INTO admins (username, password) VALUES (?, ?)',
            (username, generate_password_hash(password))
        )

    conn.commit()
    conn.close()

def verify_admin(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'SELECT id, password FROM admins WHERE username = ?',
            (username.strip(),)
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
        return False, f"Student with Roll Number '{roll_number}' already exists."
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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number.strip().upper(),))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

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
