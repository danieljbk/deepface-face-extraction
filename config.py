import os
import logging

# Constants for directories and file names
BASE_IMG_DB_NAME = "input"
CROPPED_IMG_DB_NAME = "output"
BASE_IMG_DIR_NAME = "test"
REFERENCE_IMG_FILE_PATH = os.path.join(
    BASE_IMG_DB_NAME, "reference.jpg"
)  # Faces will be compared against this reference image, which therefore must contain the individual's face at high quality.

# Constants for logging messages
FAILED_TO_FIND_FACES_LOG = "Failed to find similar faces: {img_path} due to: {error}"
FAILED_TO_LOAD_IMAGE_LOG = "Failed to load image at {img_path}; face detection aborted."
CROPPED_IMAGE_EMPTY_LOG = "Error: Cropped image is empty."
NO_FACES_SAVED_LOG = "Error: No faces saved to {directory}; operation aborted."
FOUND_SIMILAR_FACES_LOG = "Similar faces found in {directory}."
CROPPED_SAVED_FACE_LOG = "Cropped and saved face from: {img_path}"
NO_SIMILAR_FACES_FOUND_LOG = (
    "No similar faces found or an error occurred during indexing."
)


def configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )
