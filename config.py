import os
import logging

# Constants
BASE_IMG_DB_NAME = "face-db"
BASE_IMG_DIR_NAME = "chaewon"
BASE_IMG_FILE_NAME = "2023-09-28_16-00-46_UTC_8.jpg"

REFERENCE_IMG_FILE_NAME = "reference_a.jpg"  # Faces will be compared against this image, which therefore must be high quality.

CROPPED_IMG_DB_NAME = "cropped-face-db"

# Set file paths from constants
BASE_IMG_FILE_PATH = os.path.join(
    BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME, BASE_IMG_FILE_NAME
)
REFERENCE_IMG_FILE_PATH = os.path.join(
    BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME, REFERENCE_IMG_FILE_NAME
)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )
