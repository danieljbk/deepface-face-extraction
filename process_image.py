import os
import cv2
import logging
from deepface import DeepFace
from utils import load_image, save_image, ensure_directory_exists
from config import (
    BASE_IMG_FILE_PATH,
    BASE_IMG_DIR_NAME,
    CROPPED_IMG_DB_NAME,
    REFERENCE_IMG_FILE_PATH,
    configure_logging,
)

configure_logging()


def save_detected_faces(faces: list, img_path: str):
    detected_faces_dir = os.path.join(
        CROPPED_IMG_DB_NAME,
        BASE_IMG_DIR_NAME,
        os.path.basename(img_path) + "/",
    )  # the "/" is added to ensure this is recognized as a directory path and not a file
    ensure_directory_exists(detected_faces_dir)

    # load image here so I simply crop this image for every detected face
    img = load_image(img_path)
    if img is None:
        logging.error("Failed to load image, aborting face detection.")
        return None  # Exit the function if image is not loaded

    # used to create unique filename for each face (1.png, 2.png, etc.)
    count = 1
    for face in faces:
        # Extract facial area data from deepface results
        facial_area = face["facial_area"]
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )
        # Crop the image using the facial area coordinates
        cropped_img = img[y : y + h, x : x + w]
        if cropped_img.size == 0:
            logging.error("Error: Cropped image is empty.")
        else:
            # Set up the directory and filename for the cropped image
            cropped_img_filename = f"{count}.png"
            cropped_img_full_path = os.path.join(
                detected_faces_dir, cropped_img_filename
            )
            save_image(cropped_img_full_path, cropped_img)
            count += 1

    return detected_faces_dir


# Handles multiple face detections by selecting the most similar face (lowest distance value)
def find_most_similar_face(
    directory,
):
    # NOTE: DeepFace.find generates a .pkl file in the directory, allowing potential usage in subsequent runs. See CLI output example:
    # "There are now 1 representations in ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl"
    dfs = DeepFace.find(
        img_path=REFERENCE_IMG_FILE_PATH,
        db_path=directory,
        detector_backend="retinaface",
    )  # these are dataframes, ordered by most similar to least similar

    # TODO: Investigate why DeepFace.find returns this single dataframe in a list.
    dfs = dfs[0]

    # Assumes individual's face appears no more than once in photo (no collages allowed)
    try:
        # dfs could be an empty dataframe here if deepface determined that NONE of the faces were similar to the reference.
        # This is why the reference image is quite important. This can lead to complexity...
        # if there are photos of person A from a long time ago, that are compared with recent photos.
        # but, this could also be used as a feature if we can filter photos by distance and automatically group them...
        # into a batch of photos for the face recognized as "old photos of A" v.s. "recent photos of A"
        most_similar_df = dfs.loc[0]
        logging.info("Found at least 1 face similar to reference face.")

        # extract path data from dataframes
        most_similar_face_img_path = most_similar_df["identity"]

        return most_similar_face_img_path
    except IndexError as e:
        logging.error(
            "No similar faces found, or an error occurred in the DataFrame indexing."
        )
        return None
    except Exception as e:
        logging.error(f"Unexpected error when finding most similar face: {e}")
        return None


def detect_faces(img_path):
    # The most capable "backend" provided by DeepFace for face detection is "retinaface". The creator of deepface alternatively recommends "mtcnn".
    return DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="retinaface",
    )


def process_single_image(img_path):
    faces = detect_faces(img_path)
    if faces is None:
        logging.error("Failed to detect faces in image, aborting further processing.")
        return

    detected_faces_dir = save_detected_faces(faces, img_path)
    if detected_faces_dir is None:
        logging.error(
            "Failed to detect and save faces from BASE_IMG, aborting further processing."
        )
        return

    # Find the most similar face
    most_similar_face_img_path = find_most_similar_face(
        detected_faces_dir,
    )
    if most_similar_face_img_path:
        logging.info(f"Most similar face found at: {most_similar_face_img_path}")

        # Set the destination path for copying the photo
        new_most_similar_cropped_img_path = os.path.join(
            CROPPED_IMG_DB_NAME,
            BASE_IMG_DIR_NAME,
            "cropped_" + os.path.basename(img_path),
        )

        # Copy & save the most similar cropped image
        most_similar_cropped_img = load_image(most_similar_face_img_path)
        save_image(new_most_similar_cropped_img_path, most_similar_cropped_img)
        logging.info(
            f"Copied and saved the most similar face to: {new_most_similar_cropped_img_path}"
        )

    logging.info(f"Processing of image ({img_path}) complete.")


# For standalone testing of this file.
if __name__ == "__main__":
    test_img_path = BASE_IMG_FILE_PATH  # Specify a path or use an example from config
    process_single_image(test_img_path)
