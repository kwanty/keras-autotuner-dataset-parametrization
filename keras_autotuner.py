# -*- coding: utf-8 -*-
#
# Description:  Example of keras autotuner in data augmentation pipeline
# Author:       Jaroslaw Bulat (kwant@agh.edu.pl, kwanty@gmail.com)
# Created:      08.08.2021
# License:      GPL
# File:         keras_autotuner.py

import keras_tuner as kt
import tensorflow as tf
import numpy as np


def AugmentedDataset(noise_variation=0.0, batch_size=16, trainable=True):
    """
    Dataset generator, produce linear process y=a*x+b with noisy output, noise variation is a tunable parameter
    :param noise_variation: variation of noise added during augmentation
    :param batch_size: batch size
    :param trainable: True/False -> training set / validation set
    """
    dataset_size = 10000 if trainable else 1000
    x = np.arange(0, dataset_size, dtype=np.float32)    # dummy input/output
    y = x*0.1+0.2

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if trainable:   # add noise only for training dataset
        dataset = dataset.map(lambda x, y: add_noise(x, y, noise_variation))
    dataset = dataset.batch(batch_size)
    return dataset


@tf.function
def add_noise(x, y, noise_variation):
    """
    Add noise to the output (noise augmentation)
    :param x: input
    :param y: output
    :param noise_variation: desired noise variation
    :return: x+noise(noise_variation), y
    """
    # noise_variation <-- parameter set by keras tuner
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_variation)
    return x+noise, y


class MyHyperband(kt.tuners.Hyperband):
    """
    Override _build_and_fit_model() from keras_tuner.Tuner. It is called in MultiExecutionTuner
    Class hierarchy: MyRandomSearch <- RandomSearch <- MultiExecutionTuner <- Tuner
    """
    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        self.was_called = True
        noise_variation = trial.hyperparameters['aug_noise']
        batch_size = fit_kwargs['batch_size']
        train_dataset = AugmentedDataset(noise_variation=noise_variation, batch_size=batch_size, trainable=True)
        validation_dataset = AugmentedDataset(batch_size=batch_size, trainable=False)

        # TODO: menage exception during build/fit (eg. insufficient GPU RAM)
        model = self.hypermodel.build(trial.hyperparameters)
        return model.fit(train_dataset, callbacks=fit_kwargs['callbacks'], validation_data=validation_dataset)


def model_builder(hp):
    """
    Build and compile a model.
    :param hp: hyperparameters
    :return: keras model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hp['L1_size'], input_shape=[1], activation='relu'))
    model.add(tf.keras.layers.Dense(1))     # output layer
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


if __name__ == "__main__":
    # HyperParameters space
    hp = kt.HyperParameters()
    hp.Choice('L1_size', [1, 2, 4, 16, 32], default=2)  # used in model_builder(...)
    hp.Float('aug_noise', min_value=0, max_value=2, default=1)  # used in _build_and_fit_model(...)

    tuner = MyHyperband(
        hypermodel=model_builder,
        objective='val_loss',
        max_epochs=50,
        hyperparameters=hp,
        directory='./',
        project_name='keras_autotuner_dataset'
    )
    tuner.search(epochs=42, batch_size=24, verbose=True)
