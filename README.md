# face-recognition

Input:

1. Path to directory of images (not limited to images of individual)
2. Reference image of individual's face

Output:

Cropped images of individual's face from every image in the directory.

Note: Assumes individual's face appears, at maximum, once in each image.

## Purpose

Extract only the face data from a dataset of images of humans / a specific human.

## NOTE: Does not crop all photos by default (for efficiency)

After modification, code now compares each image in the directory to the reference image to see if it is a match. Then it crops the images that are a match. It used to crop all images in the directory, but this was inefficient.

The only issue this might create is that the program fully depends on the reference image for comparison, so if the reference image is not a good match to the target face, the program will miss some faces.

However, there is a way to go around this. Simply set the similarity threshold variable to 1, and the program will crop all faces it finds in the directory.

## TO-DO

### Utilize .pkl files

I should utilize the `.pkl` file, generated after each DeepFace.find(). Note that it links the reference image to the provided images - it is not just vector storage of the provided images.

Potentially related - I might want to look into the "embed" function of DeepFace, which might solve this efficiency issue.

### Filter Images by Face Dimension

Speaking of low quality, the dimensions of the cropped images communicate their quality. I might want to consider trashing any output photos that are smaller than a set number of pixels. This is important because DreamBooth wants as high quality photos as possible. Keep in mind that size does not necessarily correlate to quality. Meaning, the picture could be blurry but the size could be big. This is equally bad as the picture being clear but the size being tiny.

### Detailed Data Logging for Reuse

Need to create info files inside each image directory specifying data that can be accessed in the future to avoid reprocessing.

This includes data like:

- the reference file used
- dimensions of reference photo
- the X, Y, W, H coords of each cropped photo (relative to reference photo)
- all logs pertaining to the processing of this image
- etc.

I also need a file describing which images had an identifiable face in them, and which did not. Which had how many faces, etc.

Anything that was processed in an initial run must be saved in an easily accessible so that we do not have to waste computational resources.
