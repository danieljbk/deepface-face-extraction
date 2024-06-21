import os
import cv2
import logging


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        logging.info(f"Creating directory at: ({path})")
        os.makedirs(path)


def load_and_process_image(img_path):
    """Load an image and check if it's valid."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        logging.error(f"Failed to load image at ({img_path}).")
        return None  # Explicitly return None to signal failure
    return img


def save_processed_image(path, img):
    """Save an image and log the outcome."""
    try:
        cv2.imwrite(path, img)
        logging.info(f"Image saved successfully at ({path}).")
        return True
    except cv2.error as e:
        logging.error(f"OpenCV error when saving image at ({path}): {e}")
        return False
    except Exception as e:
        logging.error(f"General error when saving image at ({path}): {e}")
        return False
