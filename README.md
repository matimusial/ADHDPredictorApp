# ADHDPredictorApp

## Overview

This project aims to predict ADHD (Attention Deficit Hyperactivity Disorder) in children using Convolutional Neural Networks (CNNs). It leverages two primary datasets: EEG brainwave data and MRI brain images. The project comprises two main components:
1. **EEG Data Analysis**: Utilizing EEG data to predict ADHD presence in children through deep learning models.
2. **MRI Data Enhancement and Analysis**: Generating additional MRI brain images using GANs (Generative Adversarial Networks) and employing CNNs to predict ADHD based on these images.

## Datasets

### EEG Data

The EEG dataset is a collection of brainwave data from children, both with and without ADHD. This data is instrumental in training our model to identify patterns associated with ADHD.

- **Source**: [EEG Data for ADHD and Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

### MRI Data

The MRI dataset provides brain images, which are used alongside generated images to enhance the model's training and prediction capabilities.

- **Source**: [OpenNeuro Dataset ds002424](https://openneuro.org/datasets/ds002424/versions/1.2.0)

## Methodology

1. **EEG Data Processing and Model Training**: The EEG data is preprocessed to a suitable format for CNN analysis. A deep learning model is then trained to identify ADHD characteristics from the EEG patterns.
2. **MRI Image Generation and Processing**: GANs are employed to generate additional MRI images to augment the existing dataset. These images undergo preprocessing and are then used to train another CNN model for ADHD prediction.

## Objectives

- To provide an accurate and reliable method for ADHD prediction using non-invasive EEG and MRI data.
- To enhance the availability of quality MRI brain images through generative techniques, aiding in the development of robust predictive models.
