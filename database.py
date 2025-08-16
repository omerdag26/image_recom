# database.py
import sqlite3
import threading

class ImageDatabase:
    def __init__(self, db_path="images.db"):
        # connect to the SQLite database with thread safety
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # create a lock for thread-safe operations
        self._lock = threading.Lock()
        # create the images table
        self._create_table()

    # This method creates the images table if it does not exist
    def _create_table(self):
        with self._lock, self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    filepath TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    photographer TEXT
                )
            """)

    # This method adds an image to the database
    # image_id is a string, filepath is a string, width and height are integers,
    # photographer is an optional string

    def add_image(self, image_id, filepath, width, height, photographer=None):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?)",
                (str(image_id), filepath, width, height, photographer)
            )

    # This method retrieves the image ID for a given filepath
    def get_filepath(self, image_id: str) -> str | None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT filepath FROM images WHERE image_id=?",
                (str(image_id),)  # ensure TEXT key matches
            )
            row = cur.fetchone()
            return row[0] if row else None

    # This method ensures that the database connection is closed properly
    def close(self):
        self.conn.close()