import os

import tensorflow as tf
import tensorflow.keras as tf_kr

from models.BaseModel import BaseModel

class FinalModel(BaseModel):
    def set_fit_args(self):
        self.epochs = 300
        self.batch_size = 250#self.input_shape[0]
        self.metrics = ['mean_absolute_error','mean_squared_logarithmic_error','cosine_similarity','logcosh', 'accuracy']
    
    def set_model_info(self, model_name):
        self.model_name = model_name
        # self.optimizer_name = "SGD"
        # self.kernel_init_name = "glorot_uniform"
        self.activation_name = "relu"
        self.learning_rate = 0.05
    
    def init_model(self):
        # optimizer = tf_kr.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov, decay=self.decay)
        optimizer = tf_kr.optimizers.Adam(learning_rate = self.learning_rate)

        self.model.add( tf_kr.layers.InputLayer(input_shape=(self.input_shape[1],), name="input_1" ) )

        self.model.add( tf_kr.layers.Dense(8, activation=self.activation_name) )
        #self.model.add( tf_kr.layers.Dropout(0.1) )

        self.model.add( tf_kr.layers.Dense(32, activation=self.activation_name) )
        #self.model.add( tf_kr.layers.Dropout(0.1) )

        self.model.add( tf_kr.layers.Dense(32, activation=self.activation_name) )
        #self.model.add( tf_kr.layers.Dropout(0.1) )

        self.model.add( tf_kr.layers.Dense(16, activation=self.activation_name) )
        #self.model.add( tf_kr.layers.Dropout(0.1) )

        self.model.add( tf_kr.layers.Dense(1) )

        self.model.compile(
            optimizer = optimizer, 
            # (power=3, l1=0.00, l2=100, y_pred_l2=0.00) and 
            # (power=2, l1=0.00, l2=0.1, y_pred_l2=0.00) are the best. Plots are similar. Inspect these two configuration on different experiments. 
            # y_pred_l2 is not that effective as l2.
            loss = self.exponential_loss_w_elastic_reg(power=2, l1=0.00, l2=0.1, y_pred_l2=0.00),
            metrics = self.metrics)

    def fit_model(self, X_train, y_train, verbose=False):
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=False, verbose=verbose)
        return self.history