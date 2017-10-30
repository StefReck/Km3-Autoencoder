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

    logs
{'batch': 1, 'size': 10, 'loss': 1.9699528}
"""
            
class NBatchLogger_Epoch(Callback):
    #Gibt lOSS aus über alle :display batches, gemittelt über alle batches dieser epoche (das ist das was unter der verbose bar steht)
    def __init__(self, display, logfile):
        self.seen = 0
        self.display = display
        self.average = 0
        self.logfile = logfile
        self.logfile.write("#Batch\tLoss")
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



#Veraltet:
class NBatchLogger_Recent(Callback):
    #Gibt lOSS aus über alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display):
        self.seen = 0
        self.display = display
        self.average = 0

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        self.average += logs.get('loss')
        if self.seen % (self.display*logs.get('size', 0)) == 0:
            averaged_loss = self.average / (self.display)
            print ('\nSample {0}/{1} - Batch Loss: {2}'.format(self.seen,self.params['samples'], averaged_loss))
            self.average = 0

