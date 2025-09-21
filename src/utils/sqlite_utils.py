import sqlite3


def create_connection():
    conn = sqlite3.connect("indexed_docs")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS indexed_files (
                 file_name TEXT)''')
    
    return conn

def drop_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS indexed_files")
    conn.close()
    
    return conn

def add_indexed_file(file_name):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO indexed_files (file_name) VALUES (?)",
                   [file_name])
    conn.commit()
    conn.close()


def get_indexed_files():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM indexed_files")
    rows = cursor.fetchall()
    indexed_files = [row[0] for row in rows]
    conn.close()

    return indexed_files
