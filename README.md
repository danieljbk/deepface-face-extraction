# face-recognition

Takes in images of humans, outputs only the faces.
To be used to clean up the image dataset before feeding to DreamBooth.

## Installation Requirements (For Jupyter Notebook)

Before running this notebook, ensure the following packages are installed by running these commands in your command line interface (CLI):

```bash
pip install --upgrade pip setuptools wheel
pip install --upgrade pip

pip install deepface
pip install tf-keras
pip install matplotlib
```

## TO-DO:

### #2

I need to incorporate the confidence/distance attribute.

I want to input a bunch of images, in which I want the code to classify the images that will be useful versus trash ones.

To achieve this, the code will have to check each input image to see if the desired face is in there.

I should decide whether to use the `DeepFace.find` function to check against all photos BEFORE cropping them, or to just do it on-the-go while I crop them, as sometimes DeepFace seems to fail to crop when a face is not detected. This second method might be more efficient, as the first method could lead to doing double the work - identify first when checking, the identify again while cropping. Though, if I can utilize the `.pkl` file, which seems to contain info about the images, I could save compute. I might want to look into the "embed" function, which might solve this efficiency issue.

It could help to use the distance attribute to filter out low quality images if distance does increase with lower quality.

### #3

Speaking of low quality, the dimensions of the cropped images communicate their quality. I might want to consider trashing any output photos that are smaller than a set number of pixels. This is important because DreamBooth wants as high quality photos as possible. Keep in mind that size does not necessarily correlate to quality. Meaning, the picture could be blurry but the size could be big. This is equally bad as the picture being clear but the size being tiny.

### #4

Copied from detect_multiple_faces.py.
Alternative method of selecting similar faces from recognized faces in photo.

```python
# TO-DO: Edge Case (skipped for now bc unnecessary)
# if the photo was a collage of multiple photos of the specific individual
# select all faces that had a similarity distance of less than 0.5

similar_dfs = dfs[dfs["distance"] < 0.5]
file_paths = similar_dfs["identity"].tolist()
```

### #5

Need to create info files inside each image directory specifying data that can be accessed in the future to avoid reprocessing.

This includes data like:

- the reference file used
- dimensions of reference photo
- the X, Y, W, H coords of each cropped photo (relative to reference photo)
- all logs pertaining to the processing of this image
- etc.

I also need a file describing which images had an identifiable face in them, and which did not. Which had how many faces, etc.

Anything that was processed in an initial run must be saved in an easily accessible so that we do not have to waste computational resources.

### #6

Currently, when I run the "find" function, it checks all images in the given directory for faces. The problem here is that it is rechecking these cropped face images for faces.
I already cropped all of the faces in the directory, so it really shouldn't need to check all of the images for faces in the directory again.
I need it to only check the reference image for a face, or I need it to output the face coordinates for each image in the directory, which I can use to crop the images.
