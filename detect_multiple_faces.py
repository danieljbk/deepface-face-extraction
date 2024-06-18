import os
import cv2
from deepface import DeepFace
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)

# Constants
IMG_DB_NAME = "face-db"
IMG_DIR_NAME = "chaewon-test"
IMG_FILE_NAME = "2.jpg"
REFERENCE_IMG_FILE_NAME = "3.jpg"  # this image is what the faces are compared to & therefore must be high quality.


# returns True or False depending on whether image was saved
def save_image(img_destination_path: str, img):
    success = cv2.imwrite(img_destination_path, img)
    if success:
        logging.info(f"Saved photo to {img_destination_path}")
    else:
        logging.error(f"Failed to save photo to {img_destination_path}")


def load_image(full_img_path):
    img = cv2.imread(full_img_path)
    if img is None:
        logging.error(f"Error: Failed to load image from {full_img_path}.")
        return None  # Explicitly return None to signal failure
    return img


def save_detected_faces(
    faces: list, IMG_DIR_NAME: str, IMG_FILE_NAME: str, full_img_path: str
):
    cropped_img_db_name = "cropped-face-db"
    current_cropped_faces_dir_in_cropped_img_db_path = os.path.join(
        cropped_img_db_name, IMG_DIR_NAME, IMG_FILE_NAME + "/"
    )  # the slash is added to ensure this is recognized as a directory path and not a file

    # Ensure the output directory exists before saving
    if not os.path.exists(current_cropped_faces_dir_in_cropped_img_db_path):
        os.makedirs(current_cropped_faces_dir_in_cropped_img_db_path)

    # load image here so I simply crop this image for every detected face
    img = load_image(full_img_path)
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
                current_cropped_faces_dir_in_cropped_img_db_path, cropped_img_filename
            )

            save_image(cropped_img_full_path, cropped_img)
            count += 1

    return cropped_img_db_name, current_cropped_faces_dir_in_cropped_img_db_path


# NOTE: The following considers cases where multiple faces were detected in the photo.
# If there was only 1 face (most likely scenario), it will just choose that one.
def find_most_similar_face(
    REFERENCE_IMG_FILE_NAME,
    IMG_DB_NAME,
    IMG_DIR_NAME,
    current_cropped_faces_dir_in_cropped_img_db_path,
):
    reference_img_path = os.path.join(
        IMG_DB_NAME, IMG_DIR_NAME, REFERENCE_IMG_FILE_NAME
    )

    # keep in mind DeepFace.find will generate a .pkl inside the directory.
    # Example CLI output:
    # "There are now 1 representations in ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl"
    dfs = DeepFace.find(
        img_path=reference_img_path,
        db_path=current_cropped_faces_dir_in_cropped_img_db_path,
        detector_backend="retinaface",
    )  # these are dataframes, ordered by most similar to least similar

    # for some reason, these dataframes are wrapped in a list.
    # I don't get why. The dataframes are already a list. Why would there be multiple dataframe outputs?
    dfs = dfs[0]

    # useful for most cases, where the individual will only appear once in the photo
    try:
        # dfs could be an empty dataframe here...
        # if deepface determined that NONE of the faces were similar to the reference.
        # this is why the reference is quite important.
        # and this can lead to complexity...
        # if there are photos of person A from a long time ago, that are compared with recent photos.
        # but, this could also be used as a feature...
        # if we can filter photos by distance and automatically group them...
        # into a batch of photos for the face recognized as "old photos of A" v.s. "recent photos of A"
        most_similar_df = dfs.loc[0]

        logging.info("Found at least 1 face similar to reference face.")

        # extract path data from dataframes
        most_similar_cropped_img_path = most_similar_df["identity"]

        return most_similar_cropped_img_path
    except IndexError as e:
        logging.error(
            "No similar faces found, or an error occurred in the DataFrame indexing."
        )
        return None
    except Exception as e:
        logging.error(f"Unexpected error when finding most similar face: {e}")
        return None


def save_most_similar_face(most_similar_cropped_img_path, destination_path):
    # Load the most similar cropped image from directory (copy) then save (paste)
    most_similar_cropped_img = load_image(most_similar_cropped_img_path)
    save_image(destination_path, most_similar_cropped_img)


def detect_multiple_faces(
    IMG_DB_NAME: str, IMG_DIR_NAME: str, IMG_FILE_NAME: str, full_img_path: str
):
    """
    The following list contains all available "backends" provided by deepface.
    "retinaface" is the most capable one for face detection.
    the creator of deepface also recommended "mtcnn", but "retinaface" is great so we will stick to it.
    """
    backends = [
        "opencv",
        "ssd",
        "dlib",
        "mtcnn",
        "fastmtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet",
        "centerface",
    ]

    faces = DeepFace.extract_faces(
        img_path=full_img_path,
        detector_backend="retinaface",
    )
    return faces


# Combine the directory with the base path
full_img_path = os.path.join(IMG_DB_NAME, IMG_DIR_NAME, IMG_FILE_NAME)
faces = detect_multiple_faces(IMG_DB_NAME, IMG_DIR_NAME, IMG_FILE_NAME, full_img_path)
detected_faces_info = save_detected_faces(
    faces, IMG_DIR_NAME, IMG_FILE_NAME, full_img_path
)

# Check if the function returned None (indicating an error)
if detected_faces_info is None:
    logging.error("Failed to detect and save faces, aborting further processing.")
else:
    cropped_img_db_name, current_cropped_faces_dir_in_cropped_img_db_path = (
        detected_faces_info
    )

    most_similar_cropped_img_path = find_most_similar_face(
        REFERENCE_IMG_FILE_NAME,
        IMG_DB_NAME,
        IMG_DIR_NAME,
        current_cropped_faces_dir_in_cropped_img_db_path,
    )

    if most_similar_cropped_img_path:
        # set destination path (for copying photo)
        new_most_similar_cropped_img_path = os.path.join(
            cropped_img_db_name, IMG_DIR_NAME, "cropped_" + IMG_FILE_NAME
        )

        save_most_similar_face(
            most_similar_cropped_img_path, new_most_similar_cropped_img_path
        )

logging.info("Code executed without crashing.")
