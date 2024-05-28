import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from MRI.image_preprocessing import trim_rows, normalize, check_dimensions
from MRI.file_io import read_pickle, save_pickle, prepare_for_cnn
from MRI.data_validation import make_predict_data
from MRI.config import CNN_EPOCHS_MRI, CNN_BATCH_SIZE_MRI, CNN_LEARNING_RATE_MRI, CNN_INPUT_SHAPE_MRI

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_cnn_model(input_shape):
    """Builds a simplified CNN model.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model (Sequential): Built CNN model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_cnn(save, real_mri_path, predict_path, model_path):
    """Trains a simplified CNN model on MRI data.

    Args:
        save (bool): Whether to save the model after training.
        real_mri_path (str): Path to the pickle files for training data.
        predict_path (str): Path to save the prediction data.
        model_path (str): Path to save the trained model.
    """
    try:
        print(f"TRENING CNN ROZPOCZETY na {CNN_EPOCHS_MRI} EPOK...")
        print("\n")

        try:
            ADHD_DATA = read_pickle(os.path.join(real_mri_path, "ADHD_REAL.pkl"))
            CONTROL_DATA = read_pickle(os.path.join(real_mri_path, "CONTROL_REAL.pkl"))

        except Exception as e:
            print(f"Błąd w ładowaniu plików oryginalnych: {e}")
            return

        try:
            ADHD_TRIMMED = trim_rows(ADHD_DATA)
            check_dimensions(ADHD_TRIMMED)
            ADHD_NORMALIZED = normalize(ADHD_TRIMMED)

            CONTROL_TRIMMED = trim_rows(CONTROL_DATA)
            check_dimensions(CONTROL_TRIMMED)
            CONTROL_NORMALIZED = normalize(CONTROL_TRIMMED)
        except Exception as e:
            print(f"Błąd w przetwarzaniu obrazów: {e}")
            return

        X_pred, y_pred, ADHD_UPDATED, CONTROL_UPDATED = make_predict_data(ADHD_NORMALIZED, CONTROL_NORMALIZED)
        X_train, X_test, y_train, y_test = prepare_for_cnn(ADHD_UPDATED, CONTROL_UPDATED)

        model = build_cnn_model(CNN_INPUT_SHAPE_MRI)

        model.compile(optimizer=Adam(learning_rate=CNN_LEARNING_RATE_MRI), loss='binary_crossentropy',
                      metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)

        _ = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=CNN_EPOCHS_MRI,
                      batch_size=CNN_BATCH_SIZE_MRI,
                      callbacks=[reduce_lr],
                      verbose=1)

        _, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Dokładność testowa: {round(test_accuracy, 4)}")

        if save:
            model.save(os.path.join(model_path, f'{round(test_accuracy, 4)}.keras'))
            save_pickle(os.path.join(predict_path, f"X_pred_{round(test_accuracy, 4)}.pkl"), X_pred)
            save_pickle(os.path.join(predict_path, f"y_pred_{round(test_accuracy, 4)}.pkl"), y_pred)

    except Exception as e:
        print(f"Błąd podczas treningu CNN: {e}")
        return
