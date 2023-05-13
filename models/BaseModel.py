import logging
import os
import tensorflow.keras as tf_kr
import tensorflow.keras.backend as K

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
    
    def exponential_loss_w_elastic_reg(self, power=2, l1=0.00, l2=0.01, y_pred_l2=0.00):
        # default power is 2 and no l2 regularization which is ridge regression
        def loss(y_true, y_pred):
            # compute mean error raised to the power specified by the power argument
            error = K.abs(y_pred - y_true)
            mean_error = K.mean(K.pow(error, power), axis=-1)
            
            # add Elastic regularization term
            # l1_norm = K.sum(K.abs(y_pred))
            # l2_norm = K.sum(K.square(y_pred))
            l1_norm = sum([K.sum(K.abs(w)) for w in self.get_model().trainable_weights])
            l2_norm = sum([K.sum(K.square(w)) for w in self.get_model().trainable_weights])
            elastic_term = l1 * l1_norm + l2 * l2_norm
            
            y_pred_l2_norm = K.sum(K.square(y_pred))
            y_reg = y_pred_l2 * y_pred_l2_norm
            
            return mean_error + elastic_term + y_reg
        return loss