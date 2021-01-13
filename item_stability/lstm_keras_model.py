import numpy as np
import random
from keras.layers import *
import tensorflow as tf
from keras.optimizers import Adam
from keras import Model
import datetime
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile

from item_stability.data_builder import DataGen


class LSTMModel:

    def __init__(self,
                 img_dim=(128, 128, 3),
                 n_lstm=64,
                 learning_rate=5e-4,
                 epochs=15,
                 opt_batch=128,
                 output_dim=3,
                 target='lin_v',
                 target_frame=10,
                 n_frames=3,
                 datadir=""
                 ):
        self.set_random_seeds()
        self.input_shape = img_dim
        self.data_gen = DataGen(
            img_dim=img_dim,
            target=target,
            target_frame=target_frame,
            n_frames=n_frames,
            datadir=datadir
        )
        self.target = target
        self.n_frames = n_frames
        self.n_lstm = n_lstm
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.opt_batch = opt_batch
        self.loss_history = None
        self.init_session()
        self.build_model()

    def set_random_seeds(self, seed=437527822):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

    def build_model(self):
        img = Input(name="input", shape=(None,) + self.input_shape, dtype='float32')
        # x = ResNet50(include_top=False,
        #                       weights=None,
        #                       #input_tensor=x,
        #                       input_shape=(self.input_shape),
        #                       pooling="max")(img)
        x = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"))(img)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))(x)
        x = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))(x)
        x = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))(x)
        #x = TimeDistributed(SpatialDropout2D(0.2))(x)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(self.n_lstm)(x)
        x = Dropout(0.5)(x)
        self.output = Dense(self.output_dim)(x)
        self.model = Model(inputs=img, outputs=self.output)
        self.model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate)
        )

    def train(self):
        self.loss_history = self.model.fit(
            x=self.data_gen.x_train,
            y=self.data_gen.y_train,
            batch_size=self.opt_batch,
            epochs=self.epochs,
            validation_data=(self.data_gen.x_val, self.data_gen.y_val)
        )

    def init_session(self):
        config = tf.ConfigProto(
            allow_soft_placement=True
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()

    def save(self):
        import pickle
        with open("saved_runs/lstm_model_{}_step_{}_{}.pkl".format(
                self.n_frames,
                self.target,
                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), "wb") as f:
            pickle.dump(self.loss_history.history, f)
