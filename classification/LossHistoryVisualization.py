import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras

class LossHistoryVisualisation(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
        
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.val_accs = []
        self.train_losses = []
        self.train_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.train_losses.append(logs.get('loss'))
        self.val_accs.append(logs.get('val_acc'))
        self.train_accs.append(logs.get('acc'))
        plt.plot(self.val_accs)
        plt.title('Epoch accuracy on validation set')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.grid(True)
        plt.savefig(self.filename)