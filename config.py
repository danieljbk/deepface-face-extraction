import os
import logging

# Constants for directories and file names
BASE_IMG_DB_NAME = "face-db"
BASE_IMG_DIR_NAME = "chaewon-test"
BASE_IMG_FILE_NAME = ""
REFERENCE_IMG_FILE_NAME = "3.jpg"  # Faces will be compared against this image, which therefore must be high quality.
CROPPED_IMG_DB_NAME = "cropped-face-db"

# Set file paths from constants
BASE_IMG_FILE_PATH = os.path.join(
    BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME, BASE_IMG_FILE_NAME
)
REFERENCE_IMG_FILE_PATH = os.path.join(
    BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME, REFERENCE_IMG_FILE_NAME
)

# Constants for logging messages
FAILED_TO_LOAD_IMAGE_LOG = "Failed to load image at {img_path}; face detection aborted."
CROPPED_IMAGE_EMPTY_LOG = "Error: Cropped image is empty."
NO_FACES_SAVED_LOG = "Error: No faces saved to {detected_faces_dir}; operation aborted."
FOUND_SIMILAR_FACE_LOG = "Found at least one face similar to the reference image at {REFERENCE_IMG_FILE_PATH} in {directory}."
NO_SIMILAR_FACES_FOUND_LOG = (
    "No similar faces found or an error occurred during indexing."
)
UNEXPECTED_ERROR_LOG = "Unexpected error during face similarity search: {e}"
FACE_DETECTION_FAILED_LOG = "Face detection failed for {img_path}."
NO_FACES_DETECTED_LOG = "No faces detected in {img_path}; skipping."
NO_FACES_SAVED_FROM_IMAGE_LOG = "No faces saved from {img_path}; skipping."
MOST_SIMILAR_FACE_FOUND_LOG = (
    "Most similar face located at: {most_similar_face_img_path}"
)
COPIED_SAVED_FACE_LOG = (
    "Copied and saved the most similar face to {new_most_similar_cropped_img_path}."
)
NO_SIMILAR_FACES_IMAGE_LOG = "No similar faces found for {img_path}."
COMPLETED_PROCESSING_LOG = "Completed processing for {img_path}."
FAILED_TO_FIND_FACES_LOG = "Failed to find similar faces: {img_path} due to: {error}"
CROPPED_SAVED_FACE_LOG = "Cropped and saved face from: {img_path}"
FOUND_SIMILAR_FACES_LOG = "Similar faces found in {directory}."


def configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )
