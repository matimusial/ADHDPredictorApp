import os

from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from EEG.file_io import read_pickle, save_pickle, prepare_for_cnn, make_pred_data
from EEG.data_preprocessing import filter_eeg_data, clip_eeg_data, normalize_eeg_data
from CONTROLLERS.file_io import read_eeg_raw

from CONTROLLERS.metrics import WorkerMetrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_eeg_cnn_model(input_shape):
    """Builds a CNN model for EEG data.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model (Sequential): Built CNN model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (10, 1), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 1)),
        Conv2D(32, (8, 1), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 1)),
        Conv2D(64, (4, 1), activation='relu', padding='same', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 1)),
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def train_cnn_eeg(save, pickle_path, predict_path, model_path):
    """Trains a CNN model on EEG data.

    Args:
        save (bool): Whether to save the model after training.
        pickle_path (str): Path to the EEG data.
        predict_path (str): Path to save validation data.
        model_path (str): Path to save the trained model.
    """
    from EEG.config import CNN_INPUT_SHAPE, CNN_LEARNING_RATE, CNN_EPOCHS, CNN_BATCH_SIZE
    try:
        print(f"CNN TRAINING STARTED for {CNN_EPOCHS} EPOCHS...")
        print("\n")

        try:
            ADHD_DATA = read_pickle(os.path.join(pickle_path, "ADHD_EEG_DATA.pkl"))
            CONTROL_DATA = read_pickle(os.path.join(pickle_path, "CONTROL_EEG_DATA.pkl"))
        except Exception as e:
            print(f"Error loading EEG files: {e}")
            print("Did you download the files from the link in the folder EEG/TRAIN/TRAIN_DATA?")
            return

        try:
            ADHD_UPDATED, CONTROL_UPDATED, X_pred, y_pred = make_pred_data(ADHD_DATA, CONTROL_DATA)
            ADHD_FILTERED, CONTROL_FILTERED = filter_eeg_data(ADHD_UPDATED, CONTROL_UPDATED)
            ADHD_CLIPPED, CONTROL_CLIPPED = clip_eeg_data(ADHD_FILTERED, CONTROL_FILTERED)
            ADHD_NORMALIZED, CONTROL_NORMALIZED = normalize_eeg_data(ADHD_CLIPPED, CONTROL_CLIPPED)
            X_train, y_train, X_test, y_test = prepare_for_cnn(ADHD_NORMALIZED, CONTROL_NORMALIZED)
        except Exception as e:
            print(f"Error processing EEG data: {e}")
            return

        model = build_eeg_cnn_model(CNN_INPUT_SHAPE)

        optimizer = Adam(learning_rate=CNN_LEARNING_RATE)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)

        _ = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=CNN_EPOCHS,
                      batch_size=CNN_BATCH_SIZE,
                      callbacks=[reduce_lr],
                      verbose=1)

        _, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {round(final_accuracy, 4)}")

        if save:
            model.save(os.path.join(model_path, f'{round(final_accuracy, 4)}.keras'))
            # save_pickle(os.path.join(predict_path, f"X_pred_{round(final_accuracy, 4)}.pkl"), X_pred)
            # save_pickle(os.path.join(predict_path, f"y_pred_{round(final_accuracy, 4)}.pkl"), y_pred)
        return round(final_accuracy, 4)
    except Exception as e:
        print(f"Error during CNN training: {e}")
        return

def train_cnn_eeg_readraw(save, folderPath, predict_path, model_path):
    """Trains a CNN model on EEG data.

    Args:
        save (bool): Whether to save the model after training.
        folderPath (str): Path to the EEG data.
        predict_path (str): Path to save validation data.
        model_path (str): Path to save the trained model.
    """
    from EEG.config import CNN_INPUT_SHAPE, CNN_LEARNING_RATE, CNN_EPOCHS, CNN_BATCH_SIZE
    try:
        print(f"CNN TRAINING STARTED for {CNN_EPOCHS} EPOCHS...")
        print("\n")

        try:
            ADHD_DATA, CONTROL_DATA = read_eeg_raw(folderPath)
        except Exception as e:
            print(f"Error loading EEG files: {e}")
            return

        try:
            ADHD_UPDATED, CONTROL_UPDATED, X_pred, y_pred = make_pred_data(ADHD_DATA, CONTROL_DATA)
            if(len(ADHD_UPDATED) <= 1 or len(CONTROL_UPDATED) <= 1):
                ADHD_FILTERED, CONTROL_FILTERED = filter_eeg_data(ADHD_DATA, CONTROL_DATA)
            else:
                ADHD_FILTERED, CONTROL_FILTERED = filter_eeg_data(ADHD_UPDATED, CONTROL_UPDATED)
            ADHD_CLIPPED, CONTROL_CLIPPED = clip_eeg_data(ADHD_FILTERED, CONTROL_FILTERED)
            ADHD_NORMALIZED, CONTROL_NORMALIZED = normalize_eeg_data(ADHD_CLIPPED, CONTROL_CLIPPED)
            X_train, y_train, X_test, y_test = prepare_for_cnn(ADHD_NORMALIZED, CONTROL_NORMALIZED)
        except Exception as e:
            print(f"Error processing EEG data: {e}")
            return

        model = build_eeg_cnn_model(CNN_INPUT_SHAPE)

        optimizer = Adam(learning_rate=CNN_LEARNING_RATE)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)

        worker_metrics = WorkerMetrics(total_epochs=CNN_EPOCHS)

        _ = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=CNN_EPOCHS,
                      batch_size=CNN_BATCH_SIZE,
                      callbacks=[reduce_lr,worker_metrics],
                      verbose=1)

        _, final_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {round(final_accuracy, 4)}")

        if save:
            model.save(os.path.join(model_path, f'{round(final_accuracy, 4)}.keras'))
            # save_pickle(os.path.join(predict_path, f"X_pred_{round(final_accuracy, 4)}.pkl"), X_pred)
            # save_pickle(os.path.join(predict_path, f"y_pred_{round(final_accuracy, 4)}.pkl"), y_pred)
        return round(final_accuracy, 4)

    except Exception as e:
        print(f"Error during CNN training: {e}")
        return
