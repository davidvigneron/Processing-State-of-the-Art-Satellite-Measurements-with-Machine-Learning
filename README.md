# Processing State-of-the-Art Satellite Measurements with Machine Learning
## Setup
All these scripts were made for use on HoreKa (therefore also the header in most files). In ``requirements.txt`` is what I believe to be necessary to run all models. I added minimum versions, these are the versions I used, I cannot guarantee forward compability. Don't forget to adapt the venv names and enter your email, as well as the paths in ``path_constants.py``.

## Converting data to a numpy array
In ``convert_to_numpy.py`` is a script reading from LSDF and writing all data from a year into a single numpy memmap object. Using memmap because this is a very slow process and it does not make sense to use a large node for this activity. In ``filter_mmap.py`` the data is being filtered and written to a binary file. This file format is more practical, but requires loading the entire array into memory once. These binary files can still be memory mapped later, but there is no need to specify the shape and dtype of the saved values.

## Training the models
Each model was trained with the respective script. The best found hyperparameters for each model are hardcoded into the files. Code for hyperparameter search can be made available at reasonable request.

## Making predictions
In the file ``eval_one_orbit.py`` are short examples on how to make a predictions with each model.

## Acknowledgement
This work is part of my bachelor thesis at KIT with the Institut KI in den Klima- und Umweltwissenschaften and IMK-ASF.