import numpy as np
import pickle

def get_best_model_epoch(hist_filename):
    '''
    returns the epoch number of best model.
    'history' is a dictionary of accuracy and loss values as output by the keras sequential model fit function
    '''
    #load history file
    history = pickle.load(open(hist_filename, "rb"))

    #get row with highest validation accuracy - that's all :)
    valid_acc_list =  history['val_acc']
    valid_acc_array = np.array(valid_acc_list)
    return valid_acc_array.argmax()

    
