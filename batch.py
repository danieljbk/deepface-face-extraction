from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import cv2

# Define the image path and filename
img_directory = "chaewon-test"
img_filename = "multiple.jpg"

# Combine the directory with the base path
full_img_path = os.path.join("face-db", img_directory, img_filename)

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

# Load the image from the predefined path
img = cv2.imread(full_img_path)

if img is None:
    print("Failed to load image.")
    # To-do: Exit or handle error appropriately
else:
    cropped_img_db_name = "cropped-face-db"
    cropped_img_db_path = os.path.join(
        cropped_img_db_name, img_directory, img_filename + "/"
    )  # the slash is added to ensure this is recognized as a directory path and not a file

    # Ensure the output directory exists before saving
    if not os.path.exists(cropped_img_db_path):
        os.makedirs(cropped_img_db_path)

    # used later to distinguish images if there was more than 1 face
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
            print("Error: Cropped image is empty.")
        else:
            # Set up the directory and filename for the cropped image
            cropped_img_filename = f"{count}.png"
            cropped_img_full_path = os.path.join(
                cropped_img_db_path, cropped_img_filename
            )

            # Save the cropped image
            success = cv2.imwrite(cropped_img_full_path, cropped_img)
            if success:
                print("Image successfully saved.")
                count += 1
            else:
                print("Failed to save image.")

    # the following takes care of the case where there were multiple faces in the photo.
    # even if there was only one face, it will just choose that one. no problem.

    # this image is what the faces are compared to. this image must be good quality.
    reference_img_path = os.path.join("face-db", img_directory, "3.jpg")

    # these are dataframes.
    # ordered by most similar to least similar
    dfs = DeepFace.find(
        img_path=reference_img_path,
        db_path=cropped_img_db_path,
        detector_backend="retinaface",
    )

    prune_option = "single"

    # either select the single-most similar face...
    if prune_option == "single":
        # for some reason, the dataframes are wrapped in a list.
        dfs = dfs[0]

        # useful for most cases, where the individual will only appear once in the photo
        most_similar_df = dfs.loc[0]
        most_similar_cropped_img_path = most_similar_df["identity"]

        new_most_similar_cropped_img_path = os.path.join(
            cropped_img_db_name, img_directory, "cropped_" + img_filename
        )

        # Load the image from the predefined path
        most_similar_cropped_img = cv2.imread(most_similar_cropped_img_path)

        # Save the cropped image
        success = cv2.imwrite(
            new_most_similar_cropped_img_path, most_similar_cropped_img
        )
        if success:
            print("Image successfully saved.")
            count += 1
        else:
            print("Failed to save image.")

    # ...or select all faces that had a similarity distance of less than 0.5
    elif prune_option == "multiple":
        # useful if the photo was a collage of multiple photos of the specific individual
        similar_dfs = dfs[dfs["distance"] < 0.5]
        file_paths = similar_dfs["identity"].tolist()

        # TO-DO (skipped for now bc unnecessary)
