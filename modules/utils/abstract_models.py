import numpy as np


class _AbstractAutoencoder:
    """Protected class storing utility methods inherited by Autoencoders
    """
    def fit(self, X_train, epochs, batch_size, noise=None, **kwargs):
        """Method for fitting the encoder_decoder model and saving the entire
        architecture as well as the encoder, decoder parts

        Args:
            - X_train:      is a numpy array storing the values passed to the
                            autoencoder.
            - epochs:       is an integer specifyinng for how many number of
                            epochs the model will be trained
            - batch_size:   is a float and specifying the ratio of X_train
                            over which the error is computed

        Returns:
            - None
        """
        if noise is not None:
            X_target = X_train + np.random.normal(0, noise, size=X_train.shape)
        else:
            X_target = X_train
        self._encoder_decoder.fit(
            X_train,
            X_target,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        return self._encoder_decoder, self._encoder

    def encode_decode(self, X_test):
        """Method for transforming and reducing an input to size of the latent
        space of the encoder and then back to the original input, this can be
        use for value inputation in case of corrupted entries

        Args:
            - X_test:   is a numpy array storing the values that needs
                        to be transformed

        Returns:
            - a numpy array storing the values for the transformation of X_test
        """
        return self._encoder_decoder.predict(X_test)

    def encode(self, X_test):
        """Method for transforming and reducing an input to size of the latent
        space of the encoder

        Args:
            - X_test:   is a numpy array storing the values that needs
                        to be transformed

        Returns:
            - a numpy array storing the values for the transformation of X_test
        """
        return self._encoder.predict(X_test)

    def get_model(self):
        """Method for getting the stored autoencoder model.

        Returns:
            - model: a keras model, the stored autoencoder model.
        """
        model = self._encoder_decoder
        return model

    def get_model_tag(self):
        """Method for getting the model tag (identifier).

        Returns:
            -model_tag: a string specifying the model tag.
        """
        model_tag = self.model_tag
        return model_tag


class _AbstractDEC:
    """Protected class implementing methods inherited by the actual DEC models
    """
    def _print_status(self, status):
        """Protected method for printing the status of the training process

        Args:
            - status: is a dictionary, keys are logs to keep track of values
              are the metric associated to the logs

        Return:
            - None
        """
        for key, value in status.items():

            print('{}: {}'.format(key, value))

        return None

    def get_model(self):
        """Method for getting the stored IDEC model.

        Returns:
            - model: a keras model, the stored IDEC model.
        """
        model = self._model
        return model

    def set_model(self, model):
        setattr(self, '_model', model)
        setattr(self, 'n_parameters', model.count_params())
        return None

    def get_model_tag(self):
        """Method for getting the model tag (identifier).

        Returns:
            -model_tag: a string specifying the model tag.
        """
        model_tag = self.model_tag
        return model_tag
