# face-recognition

Takes in images of humans, outputs only the faces.
To be used to clean up the image dataset before feeding to DreamBooth.

## Installation Requirements (WIP)

Before running this notebook, ensure the following packages are installed by running these commands in your command line interface (CLI):

```bash
pip install --upgrade pip setuptools wheel
pip install --upgrade pip

pip install deepface
pip install tf-keras
pip install matplotlib
```

## TO-DO:

Currently, the code feeds in a single image, then faces are extracted from it.

I need to be able to feed in a directory, and then it runs the current process for all images.

I'll also need to do some clean-up work to ensure output structure.
