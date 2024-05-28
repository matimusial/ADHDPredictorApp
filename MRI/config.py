"""CONFIG FOR MRI PART"""

MODEL_CNN_NAME = "0.4041"
CNN_EPOCHS_MRI = 5
CNN_BATCH_SIZE_MRI = 32
CNN_INPUT_SHAPE_MRI = (120, 120, 1)
CNN_SINGLE_INPUT_SHAPE_MRI = 120
CNN_LEARNING_RATE_MRI = 0.001

GAN_EPOCHS_MRI = 150000
GAN_BATCH_SIZE_MRI = 32
GAN_INPUT_SHAPE_MRI = (120, 120, 1)
GAN_LEARNING_RATE = 0.0002

GENERATE_GAN_DISP_INTERVAL = 100
TRAIN_GAN_PRINT_INTERVAL = 1500
TRAIN_GAN_DISP_INTERVAL = 15000


def set_mri_epochs(new_value):
    global CNN_EPOCHS_MRI
    CNN_EPOCHS_MRI = new_value


def set_mri_batch_size(new_value):
    global CNN_BATCH_SIZE_MRI
    CNN_BATCH_SIZE_MRI = new_value


def set_mri_learning_rate(new_value):
    global CNN_LEARNING_RATE_MRI
    CNN_LEARNING_RATE_MRI = new_value


def set_mri_input_shape(new_value):
    global CNN_SINGLE_INPUT_SHAPE_MRI
    CNN_SINGLE_INPUT_SHAPE_MRI = new_value


def set_gan_epochs(new_value):
    global GAN_EPOCHS_MRI
    GAN_EPOCHS_MRI = new_value


def set_gan_batch_size(new_value):
    global GAN_BATCH_SIZE_MRI
    GAN_BATCH_SIZE_MRI = new_value


def set_gan_learning_rate(new_value):
    global GAN_LEARNING_RATE
    GAN_LEARNING_RATE = new_value


def set_gan_input_shape(new_value):
    global GAN_INPUT_SHAPE_MRI
    GAN_INPUT_SHAPE_MRI = new_value
