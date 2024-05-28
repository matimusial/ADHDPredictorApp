"""CONFIG FOR EEG PART"""

EEG_SIGNAL_FRAME_SIZE = 128
EEG_NUM_OF_ELECTRODES = 19
FS = 128

CUTOFFS = [(4,8), (12,30), (4,30)]  # Frequency [theta, beta, both]

CNN_INPUT_SHAPE = (EEG_NUM_OF_ELECTRODES, EEG_SIGNAL_FRAME_SIZE, 1)
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
MODEL_CNN_NAME = "0.9307"


def set_cnn_model_name(new_value):
    global MODEL_CNN_NAME
    MODEL_CNN_NAME = new_value


def set_cnn_epochs(new_value):
    global CNN_EPOCHS
    CNN_EPOCHS = new_value


def set_cnn_batch_size(new_value):
    global CNN_BATCH_SIZE
    CNN_BATCH_SIZE = new_value


def set_learning_rate(new_value):
    global CNN_LEARNING_RATE
    CNN_LEARNING_RATE = new_value


def set_electrodes(new_value):
    global EEG_NUM_OF_ELECTRODES
    EEG_NUM_OF_ELECTRODES = new_value


def set_frame(new_value):
    global EEG_SIGNAL_FRAME_SIZE
    EEG_SIGNAL_FRAME_SIZE = new_value


def set_fs(new_value):
    global FS
    FS = new_value
