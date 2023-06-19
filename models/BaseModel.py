import logging
import os
import tensorflow as tf
import tensorflow.keras as tf_kr
import tensorflow.keras.backend as K
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

logger = logging.getLogger(__name__)

class BaseModel():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = tf_kr.Sequential()

    def get_model(self):
        return self.model

    def get_model_info(self):
        raise NotImplementedError
    
    #Set model.fit args in child.
    def set_fit_args(self):
        raise NotImplementedError

    #This is for logging
    def set_model_info(self):
        raise NotImplementedError

    #Define model here in child classes
    def init_model(self):
        raise NotImplementedError

    # def is_fit_model_implemented(self):
    #     return False

    #Fits the model. It is for overriding input output shape etc.
    def fit_model(self):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)
    
    def exponential_loss_w_elastic_reg(self, power=2, l1=0.00, l2=0.00, reg_term="W"):
        # default power is 2 and no l2 regularization which is ridge regression
        def loss(y_true, y_pred):
            # compute mean error raised to the power specified by the power argument
            error = K.abs(y_pred - y_true)
            mean_error = K.mean(K.pow(error, power), axis=-1)
            
            # add Elastic regularization term
            elastic_term = 0
            # trainable_weights even values represent weights and odd values represent biases
            if reg_term=="W":
                # select only weights
                l1_norm = sum([K.sum(K.abs(w)) for i, w in enumerate(self.get_model().trainable_weights) if i % 2 == 0])
                l2_norm = sum([K.sum(K.abs(w)) for i, w in enumerate(self.get_model().trainable_weights) if i % 2 == 0])
                elastic_term = l1 * l1_norm + l2 * l2_norm
                
            if reg_term=="b":
                # select only biases
                l1_norm = sum([K.sum(K.abs(w)) for i, w in enumerate(self.get_model().trainable_weights) if i % 2 == 1])
                l2_norm = sum([K.sum(K.abs(w)) for i, w in enumerate(self.get_model().trainable_weights) if i % 2 == 1])
                elastic_term = l1 * l1_norm + l2 * l2_norm
            
            if reg_term=="Wb":
                # select all weights and biases
                l1_norm = sum([K.sum(K.abs(w)) for w in self.get_model().trainable_weights])
                l2_norm = sum([K.sum(K.square(w)) for w in self.get_model().trainable_weights])
                elastic_term = l1 * l1_norm + l2 * l2_norm
                
            if reg_term=="WX+b":
                # select predict as regularizer
                l1_norm = K.sum(y_pred)
                l2_norm = K.sum(K.square(y_pred))
                elastic_term = l1 * l1_norm + l2 * l2_norm
            
            return mean_error + elastic_term
        return loss
    
    def rolling_stats(self, pred_tensor, window_size, stats="median"):
        # Convert pred to a 1D tensor
        pred_1d = np.squeeze(pred_tensor.numpy(), axis=-1)
        padded = np.pad(pred_1d, (window_size, 0), mode='edge')
        if stats == "median":
            return tf.constant(np.median(np.lib.stride_tricks.sliding_window_view(padded, window_size), axis=1)[:-1])
        if stats == "mean":
            return tf.constant(np.mean(np.lib.stride_tricks.sliding_window_view(padded, window_size), axis=1)[:-1])
    
    def custom_loss(self, power=3, l1=0.00, l2=0.01, y_pred_l2=0.00):
        # default power is 2 and no l2 regularization which is ridge regression
        def loss(y_true, y_pred):
            # compute mean error raised to the power specified by the power argument
            error = K.abs(K.pow(y_pred,power) - K.pow(y_true,power))
            mean_error = K.mean(error, axis=-1)
            
            # add Elastic regularization term
            # l1_norm = K.sum(K.abs(y_pred))
            # l2_norm = K.sum(K.square(y_pred))
            l1_norm = sum([K.sum(K.abs(w)) for w in self.get_model().trainable_weights])
            l2_norm = sum([K.sum(K.square(w)) for w in self.get_model().trainable_weights])
            elastic_term = l1 * l1_norm + l2 * l2_norm
            
            # shifted_tensor = K.concatenate([K.constant([[0],[0]]), y_pred[:-2]], axis=0)
            means_of_prevs = self.rolling_stats(y_pred, window_size=24, stats="median")
            differences = y_pred - means_of_prevs
            y_pred_l2_norm = K.sum(K.square(differences))
            y_reg = y_pred_l2 * y_pred_l2_norm
            
            return mean_error + elastic_term + y_reg
        return loss