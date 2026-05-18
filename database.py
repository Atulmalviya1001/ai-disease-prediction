import sqlite3

conn = sqlite3.connect("health_app.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    disease TEXT,
    risk REAL,
    result TEXT,
    precautions TEXT
)
""")

conn.commit()
