import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from dotenv import load_dotenv
load_dotenv()


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
        )

    return conn

def drop_table(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cursor.close()
    conn.close()
    
    return conn


def add_eval_data(query, context, response,
                  retrieval_time, query_to_response_time):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS eval_metrics (
        id SERIAL PRIMARY KEY,
        query TEXT,
        context JSON,
        response TEXT,
        retrieval_time REAL,
        query_to_response_time REAL,        
        request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
                   

    cursor.execute(
        "INSERT INTO eval_metrics (query, context, response," \
        "retrieval_time, query_to_response_time) VALUES (%s, %s, %s, %s, %s)",
        (query, json.dumps(context), response, retrieval_time, query_to_response_time)
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_eval_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM eval_metrics")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return rows
