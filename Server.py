from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import subprocess
import hashlib
import re
import logging

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def init_db():
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            email TEXT UNIQUE,
                            password TEXT)''')
        conn.commit()
        logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
    finally:
        conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')

        # Validate inputs
        if not name or not email or not password:
            logging.warning("Signup attempt with missing fields")
            return jsonify({"success": False, "message": "All fields are required"})

        if not validate_email(email):
            logging.warning(f"Invalid email format: {email}")
            return jsonify({"success": False, "message": "Invalid email format"})

        if len(password) < 6:
            logging.warning("Password too short")
            return jsonify({"success": False, "message": "Password must be at least 6 characters"})

        hashed_password = hash_password(password)

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                           (name, email, hashed_password))
            conn.commit()
            logging.info(f"User registered: {email}")
            return jsonify({"success": True, "message": "Signup successful!"})
        except sqlite3.IntegrityError:
            logging.warning(f"Email already registered: {email}")
            return jsonify({"success": False, "message": "Email already registered"})
        except Exception as e:
            logging.error(f"Signup error: {e}")
            return jsonify({"success": False, "message": str(e)})
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Unexpected error in signup: {e}")
        return jsonify({"success": False, "message": "An unexpected error occurred"})

@app.route('/start-streamlit', methods=['POST'])
def start_streamlit():
    try:
        subprocess.Popen(["streamlit", "run", "Project.py"])
        logging.info("Streamlit started")
        return jsonify({"success": True, "message": "Streamlit started"})
    except Exception as e:
        logging.error(f"Streamlit start error: {e}")
        return jsonify({"success": False, "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)