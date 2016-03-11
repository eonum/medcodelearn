import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
import keras
import sys
import numpy as np

# TODO: Multiple inputs are not yet working.
class GraphMonitor(keras.callbacks.Callback):
    
    def __init__(self, base_folder, task_name='', patience=10, output_names=[]):
        self.base_folder = base_folder
        self.task_name = task_name
        self.patience = patience
        self.output_names = output_names
        
    def on_train_begin(self, logs={}):
        self.max_val_acc = 0.0
        self.max_epoch = 0
        
        self.val_losses = []
        self.epochs = []
        
        self.val_accs = {}
        self.train_accs = {}
        
        for o in self.output_names:
            self.val_accs[o] = []
            self.train_accs[o] = []
            
        self.X_train = {}
        self.y_train = {}
        self.X_val = {}
        self.y_val = {}
        
        num_outlayers = len(self.model.output_order)
        for out_layer in self.output_names:
            out_index = self.model.output_order.index(out_layer)
            
            self.X_train[out_layer] = self.model.training_data[out_index]
            # TODO check if this indexing is still valid with multiple inputs
            temp = self.model.training_data[num_outlayers + out_index]
            self.y_train[out_layer] = np.empty(temp.shape[0], dtype=np.int32)
            for i in range(0, temp.shape[0]):
                self.y_train[out_layer][i] = temp[i].argsort()[::-1][0]
            
            
            self.X_val[out_layer] = self.model.validation_data[out_index]
            temp = self.model.validation_data[num_outlayers + out_index]
            print(temp.shape)
            self.y_val[out_layer] = np.empty(temp.shape[0], dtype=np.int32)
            for i in range(0, temp.shape[0]):
                self.y_val[out_layer][i] = temp[i].argsort()[::-1][0]
            

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.epochs.append(epoch)
        
        for out_layer in self.output_names:            
            X_train = self.X_train[out_layer]
            y_train = self.y_train[out_layer]
            
            X_val = self.X_val[out_layer]
            y_val = self.y_val[out_layer]
            
            acc_val = self.accuracy(X_val, y_val, out_layer)
            acc_train = self.accuracy(X_train, y_train, out_layer)
            print('train_acc: ' + str(acc_train) + ', val_acc: ' + str(acc_val))
        
            self.val_accs[out_layer].append(acc_val)
            self.train_accs[out_layer].append(acc_train)
            
            host = host_subplot(111)
            par = host.twinx()
            host.set_xlabel('epochs')
            host.set_ylabel("accuracy for layer " + out_layer)
            par.set_ylabel("overall loss")
    
            p1, = host.plot(self.epochs, self.val_accs[out_layer], label='Validation accuracy')
            p2, = host.plot(self.epochs, self.train_accs[out_layer], label='Training accuracy')
            p3, = par.plot(self.epochs, self.val_losses, label="Validation loss")
            
            leg = plt.legend(loc='lower left')
    
            host.yaxis.get_label().set_color(p1.get_color())
            leg.texts[0].set_color(p1.get_color())
            leg.texts[1].set_color(p2.get_color())
            
            par.yaxis.get_label().set_color(p3.get_color())
            leg.texts[2].set_color(p3.get_color())
            
            plt.title('Accuracy by epoch on validation set for layer ' + out_layer)
            plt.savefig(self.base_folder + self.task_name + '_epochs_' + out_layer + '.png')
            plt.close()
            
            if acc_val > self.max_val_acc:
                self.max_val_acc = acc_val
                self.max_epoch = epoch
            elif epoch - self.max_epoch > self.patience:
                print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
        
        # Do also flush STDOUT
        sys.stdout.flush()
    
    def accuracy(self, X, y, out_layer):
        probabs = self.model.predict({'codes_input':X}, verbose=0)[out_layer]
        acc = 0.0
        for i in range(0, probabs.shape[0]):
            label = probabs[i].argsort()[::-1][0]
            if label == y[i]:
                acc += 1.0
        return acc / probabs.shape[0]
