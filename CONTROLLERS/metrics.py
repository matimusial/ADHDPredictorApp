from tensorflow.keras.callbacks import Callback

# Global variables for metrics
global_accuracy = []
global_val_accuracy = []
global_loss = []
global_val_loss = []


class WorkerMetrics(Callback):
    """Callback for updating global metrics during model training."""

    def on_epoch_end(self, epoch, logs=None):
        global global_accuracy, global_val_accuracy, global_loss, global_val_loss
        logs = logs or {}
        global_accuracy.append(logs.get('accuracy'))
        global_val_accuracy.append(logs.get('val_accuracy'))
        global_loss.append(logs.get('loss'))
        global_val_loss.append(logs.get('val_loss'))

