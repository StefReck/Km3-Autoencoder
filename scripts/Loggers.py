# -*- coding: utf-8 -*-

from keras.callbacks import Callback
from os import fsync

""" 
    on batch end params:
do_validation
verbose
metrics
batch_size
steps
epochs
samples
{'do_validation': True, 'verbose': 0, 'metrics': ['loss', 'val_loss'], 'batch_size': 10, 'steps': None, 'epochs': 2, 'samples': 270}

    logs onbatchend
{'batch': 1, 'size': 10, 'loss': 1.9699528}
{'acc': 0.5, 'batch': 0, 'loss': 0.70633852, 'size': 32}
"""

#Writes loss to a logfile every display batches, averaged over all batches in this epoch
#(This is what is shown below the verbose bar)
class NBatchLogger_Epoch(Callback):
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.average = 0
        self.logfile = logfile
        self.logfile.write("#Epoch Logger\n#Batch\tLoss")
        #self.total_number_of_batches=self.params['samples']/self.params['batch_size']

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.average += logs.get('loss')
        if self.seen % (self.display) == 0:
            averaged_loss = self.average / (self.seen)
            self.logfile.write('\n{0}\t{1}'.format(self.seen, averaged_loss))
            self.logfile.flush()
            fsync(self.logfile.fileno())

    def on_epoch_end(self, epoch, logs={}):
        self.seen = 0
        self.average  = 0

#Like the above, but also logs accuracy
class NBatchLogger_Epoch_Acc(Callback):
    #Gibt lOSS aus über alle :display batches, gemittelt über alle batches dieser epoche (das ist das was unter der verbose bar steht)
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.averageLoss = 0
        self.averageAcc = 0
        self.logfile = logfile
        self.logfile.write("#Epoch Logger\n#Batch\tLoss\tAccuracy")
        #self.total_number_of_batches=self.params['samples']/self.params['batch_size']

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageAcc += logs.get("acc")
        if self.seen % (self.display) == 0:
            averaged_loss = self.averageLoss / (self.seen)
            averaged_acc = self.averageAcc / (self.seen)
            self.logfile.write('\n{0}\t{1}\t{2}'.format(self.seen, averaged_loss, averaged_acc))
            self.logfile.flush()
            fsync(self.logfile.fileno())

    def on_epoch_end(self, epoch, logs={}):
        self.seen = 0
        self.averageLoss  = 0
        self.averageAcc = 0


#Writes loss to a logfile every display batches, averaged over the last display batches
class NBatchLogger_Recent(Callback):
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.average = 0
        self.logfile = logfile
        self.logfile.write("#Recent Logger\n#Batch\tLoss")
                           
    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.average += logs.get('loss')
        if self.seen % (self.display) == 0:
            averaged_loss = self.average / (self.display)
            self.logfile.write('\n{0}\t{1}'.format(self.seen, averaged_loss))
            self.logfile.flush()
            fsync(self.logfile.fileno())
            self.average = 0

#Like the above, but also logs accuracy
class NBatchLogger_Recent_Acc(Callback):
    #Gibt lOSS aus über alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.averageLoss = 0
        self.averageAcc = 0
        self.logfile = logfile
        self.logfile.write("#Recent Logger\n#Batch\tLoss\tAccuracy")
                           
    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageAcc += logs.get("acc")
        if self.seen % (self.display) == 0:
            averaged_loss = self.averageLoss / (self.display)
            averaged_acc = self.averageAcc / (self.seen)
            self.logfile.write('\n{0}\t{1}\t{2}'.format(self.seen, averaged_loss, averaged_acc))
            self.logfile.flush()
            fsync(self.logfile.fileno())
            self.averageLoss = 0
            self.averageAcc = 0
