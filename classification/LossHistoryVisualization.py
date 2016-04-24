import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
import keras
import sys

class LossHistoryVisualisation(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
        
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.val_accs = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_output_acc'))
        self.epochs.append(epoch)
        
        host = host_subplot(111)
        par = host.twinx()
        host.set_xlabel('epochs')
        host.set_ylabel("Accuracy")
        par.set_ylabel("Loss")

        p1, = host.plot(self.epochs, self.val_accs, label='Accuracy')
        p2, = par.plot(self.epochs, self.val_losses, label="Loss")
        
        leg = plt.legend(loc='lower left')

        host.yaxis.get_label().set_color(p1.get_color())
        leg.texts[0].set_color(p1.get_color())
        
        par.yaxis.get_label().set_color(p2.get_color())
        leg.texts[1].set_color(p2.get_color())
        
        plt.title('Accuracy by epoch on validation set')
        plt.savefig(self.filename)
        plt.close()
        
        # Do also flush STDOUT
        sys.stdout.flush()