import sqlite3
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.info("Database connection established.")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database connection."""
        if self.conn:
            if exc_type:
                self.conn.rollback()
                logger.warning("Transaction rolled back due to an exception.")
            else:
                self.conn.commit()
                logger.info("Transaction committed successfully.")
            self.conn.close()
            logger.info("Database connection closed.")

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
            logger.info("Chats table ensured.")

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
            logger.info("Logs table ensured.")

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
            logger.info("LLM responses table ensured.")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def save_chat(self, session_id, query, response, context=None, tokens_used=None):
        """Save a chat record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO chats (session_id, user_query, llm_response, context, tokens_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, query, response, context, tokens_used))
            logger.info(f"Chat saved for session_id: {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Error saving chat: {e}")
            raise

    def save_log(self, session_id, log_level, message, error_stack=None, request_data=None, response_data=None):
        """Save a log record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO logs (session_id, log_level, message, error_stack, request_data, response_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, log_level, message, error_stack, request_data, response_data))
            logger.info(f"Log saved for session_id: {session_id} with level: {log_level}")
        except sqlite3.Error as e:
            logger.error(f"Error saving log: {e}")
            raise

    def save_llm_response(self, session_id, prompt, response_text, tokens_used=None, processing_time=None, model_used=None, temperature=None):
        """Save an LLM response record to the database."""
        try:
            self.cursor.execute('''
                INSERT INTO llm_responses (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, prompt, response_text, tokens_used, processing_time, model_used, temperature))
            logger.info(f"LLM response saved for session_id: {session_id}")
        except sqlite3.Error as e:
            logger.error(f"Error saving LLM response: {e}")
            raise

    def get_chats_within_timeframe(self, days=None):
        """Retrieve chats within a specific timeframe."""
        try:
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                self.cursor.execute('SELECT * FROM chats WHERE timestamp >= ?', (cutoff_date,))
            else:
                self.cursor.execute('SELECT * FROM chats')
            chats = self.cursor.fetchall()
            logger.info(f"Retrieved {len(chats)} chat records.")
            return chats
        except sqlite3.Error as e:
            logger.error(f"Error retrieving chats: {e}")
            raise

    def get_logs_within_timeframe(self, days=None):
        """Retrieve logs within a specific timeframe."""
        try:
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                self.cursor.execute('SELECT * FROM logs WHERE timestamp >= ?', (cutoff_date,))
            else:
                self.cursor.execute('SELECT * FROM logs')
            logs = self.cursor.fetchall()
            logger.info(f"Retrieved {len(logs)} log records.")
            return logs
        except sqlite3.Error as e:
            logger.error(f"Error retrieving logs: {e}")
            raise

    def get_llm_responses_within_timeframe(self, days=None):
        """Retrieve LLM responses within a specific timeframe."""
        try:
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                self.cursor.execute('SELECT * FROM llm_responses WHERE timestamp >= ?', (cutoff_date,))
            else:
                self.cursor.execute('SELECT * FROM llm_responses')
            responses = self.cursor.fetchall()
            logger.info(f"Retrieved {len(responses)} LLM response records.")
            return responses
        except sqlite3.Error as e:
            logger.error(f"Error retrieving LLM responses: {e}")
            raise

    def get_average_response_time(self):
        """Calculate the average response time from LLM responses."""
        try:
            self.cursor.execute('SELECT AVG(processing_time) FROM llm_responses')
            avg_time = self.cursor.fetchone()[0]
            logger.info(f"Average response time: {avg_time} seconds.")
            return avg_time
        except sqlite3.Error as e:
            logger.error(f"Error calculating average response time: {e}")
            raise

    def get_token_usage_by_model(self):
        """Retrieve average token usage grouped by LLM model."""
        try:
            self.cursor.execute('''
                SELECT model_used, AVG(tokens_used) 
                FROM llm_responses 
                GROUP BY model_used
            ''')
            results = self.cursor.fetchall()
            logger.info(f"Retrieved token usage data for {len(results)} models.")
            return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving token usage by model: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    with DatabaseManager() as db:
        # Initialize the database tables
        db.init_db()

        # Save a chat record
        db.save_chat(
            session_id="session123",
            query="How's the weather?",
            response="It's sunny today.",
            context="General inquiry",
            tokens_used=15
        )

        # Save a log record
        db.save_log(
            session_id="session123",
            log_level="INFO",
            message="Chat saved successfully."
        )

        # Save an LLM response
        db.save_llm_response(
            session_id="session123",
            prompt="Tell me a joke.",
            response_text="Why did the chicken cross the road?",
            tokens_used=20,
            processing_time=0.5,
            model_used="GPT-4",
            temperature=0.7
        )

        # Retrieve chats from the last 7 days
        recent_chats = db.get_chats_within_timeframe(days=7)
        print(f"Recent Chats: {recent_chats}")

        # Get average response time
        avg_time = db.get_average_response_time()
        print(f"Average Response Time: {avg_time} seconds")

        # Get token usage by model
        token_usage = db.get_token_usage_by_model()
        print(f"Token Usage by Model: {token_usage}")
