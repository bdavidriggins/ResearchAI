import sqlite3
import logging
from datetime import datetime, timedelta

# Database file name
DB_FILE = 'chat_history.db'

class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.conn = None
        self.cursor = None

    def __enter__(self):
        """Establish a database connection and cursor."""
        try:
            self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database connection."""
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()

    def init_db(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            # Create the chats table
            self.cursor.execute('''
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
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_stack TEXT,
                    request_data TEXT,
                    response_data TEXT
                )
            ''')

            # Create the llm_responses table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_responses (
                    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    tokens_used INTEGER,
                    processing_time REAL,
                    model_used TEXT,
                    temperature REAL
                )
            ''')
        except sqlite3.Error as e:
            raise

    def save_chat(self, session_id, query, response, context=None, tokens_used=None):
        """Save a chat record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO chats (session_id, user_query, llm_response, context, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, query, response, context, tokens_used))
        except sqlite3.Error as e:
            raise

    def save_log(self, session_id, log_level, message, error_stack=None, request_data=None, response_data=None):
        """Save a log record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO logs (session_id, log_level, message, error_stack, request_data, response_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, log_level, message, error_stack, request_data, response_data))
        except sqlite3.Error as e:
            raise

    def save_llm_response(self, session_id, prompt, response_text, tokens_used=None, processing_time=None, model_used=None, temperature=None):
        """Save an LLM response record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO llm_responses (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature))
        except sqlite3.Error as e:
            raise

    # ... (Other methods remain unchanged)

class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that writes logs to the SQLite database."""

    def __init__(self, db_file=DB_FILE):
        super().__init__()
        self.db_file = db_file

    def emit(self, record):
        """Write a log record to the database."""
        try:
            # Open a database connection
            with DatabaseManager(self.db_file) as db:
                # Prepare log data
                session_id = getattr(record, 'session_id', 'unknown')
                log_level = record.levelname
                message = self.format(record)
                error_stack = getattr(record, 'error_stack', None)
                request_data = getattr(record, 'request_data', None)
                response_data = getattr(record, 'response_data', None)

                # Save the log to the database
                db.save_log(
                    session_id=session_id,
                    log_level=log_level,
                    message=message,
                    error_stack=error_stack,
                    request_data=request_data,
                    response_data=response_data
                )
        except Exception as e:
            # If logging to database fails, fallback to console
            print(f"Failed to log to database: {e}")

# Configure the root logger to use DatabaseLogHandler
def configure_logging():
    """Configure logging to use the database handler."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Add the database handler
    db_handler = DatabaseLogHandler()
    logger.addHandler(db_handler)

    # Optionally, add a console handler for immediate feedback
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

# Initialize the database and configure logging when the module is imported
with DatabaseManager() as db:
    db.init_db()

configure_logging()
