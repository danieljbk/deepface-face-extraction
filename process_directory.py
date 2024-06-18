import os
import logging
from config import BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME, configure_logging
from process_image import (
    process_single_image,
)

configure_logging()


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):  # Handle JPEG images
                img_path = os.path.join(root, file)
                process_single_image(img_path)


if __name__ == "__main__":
    BASE_IMG_DIR_PATH = os.path.join(BASE_IMG_DB_NAME, BASE_IMG_DIR_NAME)

    process_directory(BASE_IMG_DIR_PATH)
    logging.info("Completed processing all images in the directory.")
