# ADHD Predictor

## Overview
This project aims to predict ADHD (Attention Deficit Hyperactivity Disorder) in children using Convolutional Neural Networks (CNN). It utilizes two main datasets: EEG brain wave data and MRI brain images. The project consists of two main components:
1. **EEG Data Analysis**: Using EEG data to predict the presence of ADHD in children with deep learning models.
2. **MRI Data Augmentation and Analysis**: Generating additional MRI brain images using Generative Adversarial Networks (GAN) and using CNN to predict ADHD from original images.

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
- Mateusz Musiał
- Jacek Lal
- Radosław Nelza
- Artur Panasiuk
