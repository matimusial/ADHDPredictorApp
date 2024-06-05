
from tensorflow.keras.callbacks import Callback


# Global variables for metrics
global_accuracy = []
global_val_accuracy = []
global_loss = []
global_val_loss = []
global_epoch_count = []

class WorkerMetrics(Callback):
    """Callback for updating global metrics during model training."""

    def on_epoch_end(self, epoch, logs=None):
        global global_accuracy, global_val_accuracy, global_loss, global_val_loss, global_epoch_count
        logs = logs or {}
        epoch_count = 1
        global_accuracy.append(logs.get('accuracy'))
        global_val_accuracy.append(logs.get('val_accuracy'))
        global_loss.append(logs.get('loss'))
        global_val_loss.append(logs.get('val_loss'))
        global_epoch_count.append(logs.get(epoch_count))


        print("\nTest: ")
        print("global_accuracy: ",global_accuracy)
        print("global_val_accuracy: ",global_val_accuracy)
        print("global_loss: ",global_loss)
        print("global_val_loss: ",global_val_loss)
        print("global_epoch_count: ",global_epoch_count,"\n")


        epoch_count += 1
