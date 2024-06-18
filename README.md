# ADHD Predictor

## Overview

This project aims to predict ADHD (Attention Deficit Hyperactivity Disorder) in children using Convolutional Neural Networks (CNN). It utilizes two main datasets: EEG brain wave data and MRI brain images. The project consists of two main components:

1. **EEG Data Analysis**: Using EEG data to predict the presence of ADHD in children with deep learning models.
2. **MRI Data Augmentation and Analysis**: Generating additional MRI brain images using Generative Adversarial Networks (GAN) and using CNN to predict ADHD from original images.

**Note:** This project is being carried out at the West Pomeranian University of Technology in Szczecin. The full version of the program with a user interface does not function fully without a connection to the database, which is only available on the university network. To use the application without the database, it can be run in console mode. You need to download the repository and run the `main_MRI` or `main_EEG` files after obtaining the necessary files from the locations specified in `link.txt`. It may be necessary to make minor modifications to the code for the program to work correctly. You can also download the [executable file](#), but without a connection to the database, the program's functionality will be limited.

The entire program specification can be found on the `info.html` page under the "Specification" section.

## Datasets

### EEG Data

The EEG dataset is a collection of brain waves from children, both with and without ADHD. This data is crucial for training our model to identify patterns associated with ADHD.

**Source**: [EEG Data for ADHD and Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

### MRI Data

The MRI dataset provides brain images, which are used along with generated images to enhance the model's training and prediction capabilities.

**Source**: [OpenNeuro Dataset ds002424](https://openneuro.org/datasets/ds002424/versions/1.2.0)

## Methodology

1. **EEG Data Processing and Model Training**: EEG data is processed into a format suitable for CNN analysis. Then, a deep learning model is trained to identify ADHD features from EEG patterns.
2. **MRI Image Generation and Processing**: GANs are used to generate additional MRI images to increase the existing dataset. These images are then processed and used to train another CNN model to predict ADHD.

## Objectives

- Provide an accurate and reliable method for predicting ADHD using non-invasive EEG and MRI data.
- Increase the availability of high-quality MRI brain images through generative techniques, supporting the development of robust predictive models.

## Contributors

&copy; Mateusz Musiał, Jacek Lal, Radosław Nelza, Artur Panasiuk
