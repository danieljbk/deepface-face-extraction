# face-recognition

Takes in images of humans, outputs only the faces.
To be used to clean up the image dataset before feeding to DreamBooth.

## Installation Requirements (For the Jupyter Notebooks)

Before running this notebook, ensure the following packages are installed by running these commands in your command line interface (CLI):

```bash
pip install --upgrade pip setuptools wheel
pip install --upgrade pip

pip install deepface
pip install tf-keras
pip install matplotlib
```

## TO-DO:

### #1

Currently, the code feeds in a single image, then faces are extracted from it.

I need to be able to feed in a directory, and then it runs the current process for all images.

I'll also need to do some clean-up work to ensure output structure.

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
