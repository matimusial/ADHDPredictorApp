"""CONFIG FOR MRI PART"""

MODEL_CNN_NAME = "0.8836"
CNN_EPOCHS_MRI = 8
CNN_BATCH_SIZE_MRI = 32
CNN_INPUT_SHAPE_MRI = (120, 120, 1)
CNN_SINGLE_INPUT_SHAPE_MRI = 120
CNN_LEARNING_RATE_MRI = 0.001
FS_MRI = 128

GAN_EPOCHS_MRI = 150000
GAN_BATCH_SIZE_MRI = 32
GAN_INPUT_SHAPE_MRI = (120, 120, 1)
GAN_LEARNING_RATE = 0.0002

GENERATE_GAN_DISP_INTERVAL = 100
TRAIN_GAN_PRINT_INTERVAL = 1500
TRAIN_GAN_DISP_INTERVAL = 15000
