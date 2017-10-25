from functools import partial

import tensorflow as tf
import sonnet as snt

from model import AIRModel
from seq_model import SeqAIRModel
from modules import BaselineMLP, Encoder, Decoder, StochasticTransformParam, StepsPredictor


class AIRonMNIST(AIRModel):
    """Implements AIR for the MNIST dataset"""

    def __init__(self, obs, nums, glimpse_size=(20, 20),
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 baseline_hidden=[256, 128]*1,
                 transform_var_bias=-2.,
                 step_bias=0.,
                 *args, **kwargs):

        self.transform_var_bias = tf.Variable(transform_var_bias, trainable=False, dtype=tf.float32,
                                                       name='transform_var_bias')
        self.step_bias = tf.Variable(step_bias, trainable=False, dtype=tf.float32, name='step_bias')
        self.baseline = BaselineMLP(baseline_hidden)

        super(AIRonMNIST, self).__init__(
            *args,
            obs=obs,
            nums=nums,
            glimpse_size=glimpse_size,
            n_appearance=50,
            transition=snt.LSTM(256),
            input_encoder=partial(Encoder, inpt_encoder_hidden),
            glimpse_encoder=partial(Encoder, glimpse_encoder_hidden),
            glimpse_decoder=partial(Decoder, glimpse_decoder_hidden),
            transform_estimator=partial(StochasticTransformParam, transform_estimator_hidden,
                                      scale_bias=self.transform_var_bias),
            steps_predictor=partial(StepsPredictor, steps_pred_hidden, self.step_bias),
            output_std=.3,
            **kwargs
        )


class SeqAIRonMNIST(SeqAIRModel):
    """Implements AIR for the MNIST dataset"""

    def __init__(self, obs, glimpse_size=(20, 20),
                 inpt_encoder_hidden=[256]*2,
                 glimpse_encoder_hidden=[256]*2,
                 glimpse_decoder_hidden=[252]*2,
                 transform_estimator_hidden=[256]*2,
                 steps_pred_hidden=[50]*1,
                 baseline_hidden=[256, 128]*1,
                 transform_var_bias=-2.,
                 step_bias=0.,
                 *args, **kwargs):

        self.transform_var_bias = tf.Variable(transform_var_bias, trainable=False, dtype=tf.float32,
                                                       name='transform_var_bias')
        self.step_bias = tf.Variable(step_bias, trainable=False, dtype=tf.float32, name='step_bias')
        self.baseline = BaselineMLP(baseline_hidden)

        super(SeqAIRonMNIST, self).__init__(
            *args,
            obs=obs,
            glimpse_size=glimpse_size,
            n_what=50,
            transition=snt.LSTM(256),
            input_encoder=partial(Encoder, inpt_encoder_hidden),
            glimpse_encoder=partial(Encoder, glimpse_encoder_hidden),
            glimpse_decoder=partial(Decoder, glimpse_decoder_hidden),
            transform_estimator=partial(StochasticTransformParam, transform_estimator_hidden,
                                      scale_bias=self.transform_var_bias),
            steps_predictor=partial(StepsPredictor, steps_pred_hidden, self.step_bias),
            output_std=.3,
            **kwargs
        )