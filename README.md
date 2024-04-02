# Pneumonia Detection with Chest DICOM X-ray Images
## Overview

This project aims to develop a machine learning model capable of identifying and localizing pneumonia in chest DICOM X-ray images.
It utilizes a classification-detection pipeline, with the classification phase serving as a filter for the detection phase.
Due to computational constraints, the architectures chosen for this project are ResNet34 & EfficientNet B3 for classification and Faster RCNN & RetinaNet for detection.
The main framework used is PyTorch, employing an Object-Oriented Programming (OOP) approach.

## Features

- **Data Exploration**: Conducts detailed exploratory data analysis (EDA) to understand the dataset's characteristics and challenges.
- **Model Development**: Implements machine learning models for the classification and detection of pneumonia.
- **Evaluation**: Utilizes metrics such as AUC (classification), mAP (detection - special variation), and more to evaluate model performance.

## Environment Setup

This project is developed and tested using PyTorch, which requires CUDA for GPU acceleration. Follow the steps below to set up your environment:

### Installing PyTorch with CUDA Support

The project uses PyTorch with CUDA 11.8 support. You can install PyTorch along with torchvision and torchaudio using the following command:
'''bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
'''
This command installs the versions of PyTorch, torchvision, and torchaudio that are compatible with CUDA 11.8. Ensure that your system has CUDA 11.8 compatible hardware and drivers.

## Checking CUDA Availability

After installing PyTorch, you can verify that CUDA is available to PyTorch with the following Python commands:

'''python
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)  # Prints the CUDA version used by PyTorch
'''

If torch.cuda.is_available() returns True, your PyTorch installation can utilize GPU acceleration.

## Additional Dependencies

Install the remaining project dependencies from the requirements.txt file:

'''bash
pip install -r requirements.txt
'''

The requirements.txt file contains all the necessary Python packages to run the project. Adjustments may be needed depending on your specific system configuration and the versions of the packages at the time of your project setup.
Note:

- **CUDA Compatibility**: The installation command provided assumes compatibility with CUDA 11.8. If your GPU or system requires a different version of CUDA, please visit the visit the [PyTorch official website](https://pytorch.org/) for the appropriate installation command tailored to your CUDA version.
- **Google Colab Users**: If you are running this project in Google Colab, CUDA and PyTorch are pre-installed, and you typically do not need to manually install CUDA. You can directly proceed with installing any additional Python packages specified in requirements.txt.

# Running the Notebook

Open the Jupyter notebook in your preferred environment (e.g., JupyterLab, classic Jupyter Notebook) and run the cells sequentially to reproduce the project's findings. Note that some sections are specifically designed for Google Colab and are marked accordingly. 
For other environments, you will need to comment out the parts of the code that are for Google Colab and uncomment those marked for Jupyter Notebook.

# Notebook Structure

- **Introduction and Objectives**
- **Exploratory Data Analysis (EDA)**
- **Model Development and Training**
- **Model Evaluation**
- **Conclusion and Future Work**

# Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

# Contact

For any questions or suggestions regarding this project, please feel free to contact me at [eli.datasci.direct@gmail.com].
