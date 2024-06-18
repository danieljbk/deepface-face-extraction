import os
import cv2
import logging


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        logging.info("Directory did not exist at {path}, attempting to create...")
        os.makedirs(path)


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        logging.error(f"Error: Failed to load image from {path}.")
        return None  # Explicitly return None to signal failure
    return img


def save_image(path: str, img):
    success = cv2.imwrite(path, img)
    if success:
        logging.info(f"Saved photo to {path}")
    else:
        logging.error(f"Failed to save photo to {path}")
