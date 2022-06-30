# Keras Autotuner for Dataset parametrization
An example of dataset parameter as an autotuner hyperparameter.

The [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) cannot handle parametrization of dataset. 
It is, however, possible by subclassing original tuner. This example is proof-of-concept of that way.

## The Problem
Let's assume we have dataset generator `AugmentedDataset(...)`. It produce augmented dataset. 
The augmentation is done by adding noise: `add_noise(...)` function.

Noise variation (strength of the augmentation) could be threaded as a hyperparameter and can be optimized by the tuner.

In the standard tuner configuration, the dataset is provided as a parameter to the `tuner.search(img_train, img_label, ...)` method.
In this way you are not able to generate dataset per each training episode. It is impossible to pass hyperparameters to the dataset.

## The Solution
You can subclass the original tuner and generate dataset per each training episode. 
In the following example, the `MyHyperband` class derived from the `kt.tuners.Hyperband` is used to manage this process.

```python
class MyHyperband(kt.tuners.Hyperband):
    """
    Override _build_and_fit_model() from keras_tuner.Tuner. It is called in MultiExecutionTunerr
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
```

For each training episode, the tuner will call `_build_and_fit_model` method. 
In this method, first the dataset is generated. The dataset takes the `noise_variation` as a parameter.
This parameter is one of the hyperparameters managed by the tuner. Then the model is built and trained.

## An example
The `keras_autotuner.py` is a working example of the above solution. It consists of a trivial model (single layer), 
dataset generating data as a function: $y=x*0.1+0.2+n$ where $n$ is a noise. The noise is added only during training.
The variation of the noise is controlled by the tuner.

Finally, there are two hiperparameters: 
- `L1_size` the number of units in the first layer of the model
- `aug_noise` noise variation added to the data during the data augmentation
