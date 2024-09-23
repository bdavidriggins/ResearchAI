import sqlite3
from datetime import datetime, timedelta

# Initialize the database and tables if they don't exist
def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    # Create the chats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT NOT NULL,
            llm_response TEXT NOT NULL,
            context TEXT,
            tokens_used INTEGER
        )
    ''')

    # Create the logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            log_level TEXT NOT NULL,  -- INFO, DEBUG, ERROR, etc.
            message TEXT NOT NULL,
            error_stack TEXT,  -- Optional: store full error stack trace
            request_data TEXT, -- Optional: store request details
            response_data TEXT -- Optional: store response details
        )
    ''')

    # Create the llm_responses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT NOT NULL,
            response_text TEXT NOT NULL,
            tokens_used INTEGER,
            processing_time REAL,  -- Response time in seconds
            model_used TEXT,       -- Which LLM model was used
            temperature REAL       -- LLM temperature setting
        )
    ''')

    conn.commit()
    conn.close()

# Save a chat record
def save_chat(session_id, query, response, context, tokens_used):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chats (session_id, user_query, llm_response, context, tokens_used)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, query, response, context, tokens_used))
    conn.commit()
    conn.close()

# Save a log record
def save_log(session_id, log_level, message, error_stack=None, request_data=None, response_data=None):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO logs (session_id, log_level, message, error_stack, request_data, response_data)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, log_level, message, error_stack, request_data, response_data))
    conn.commit()
    conn.close()

# Save an LLM response
def save_llm_response(session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO llm_responses (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature))
    conn.commit()
    conn.close()

# Retrieve chats within a time frame
def get_chats_within_timeframe(days=None):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
            SELECT * FROM chats WHERE timestamp >= ?
        ''', (cutoff_date,))
    else:
        cursor.execute('SELECT * FROM chats')

    chats = cursor.fetchall()
    conn.close()
    return chats

# Retrieve logs within a time frame
def get_logs_within_timeframe(days=None):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
            SELECT * FROM logs WHERE timestamp >= ?
        ''', (cutoff_date,))
    else:
        cursor.execute('SELECT * FROM logs')

    logs = cursor.fetchall()
    conn.close()
    return logs

# Retrieve LLM responses within a time frame
def get_llm_responses_within_timeframe(days=None):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        cursor.execute('''
            SELECT * FROM llm_responses WHERE timestamp >= ?
        ''', (cutoff_date,))
    else:
        cursor.execute('SELECT * FROM llm_responses')

    responses = cursor.fetchall()
    conn.close()
    return responses

# Additional queries for analysis
def get_average_response_time():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT AVG(processing_time) FROM llm_responses
    ''')
    avg_time = cursor.fetchone()[0]
    conn.close()
    return avg_time

def get_token_usage_by_model():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT model_used, AVG(tokens_used) FROM llm_responses GROUP BY model_used
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

