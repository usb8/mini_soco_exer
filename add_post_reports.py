import sqlite3

DB_FILE = "database.sqlite"
try:
    conn = sqlite3.connect(DB_FILE)
    print("SQLite Database connection successful")
except Exception as e:
    print(f"Error: '{e}'")

conn.execute("""
CREATE TABLE `post_reports` (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL,
    reporter_id INTEGER NOT NULL
);
""")
conn.commit()
conn.close()