from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.layers import GaussianNoise, SpatialDropout1D
from tensorflow.keras.layers import SpatialDropout2D, UpSampling2D
from tensorflow.keras.layers import LSTM, RepeatVector, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

from ..utils.abstract_models import _AbstractAutoencoder


class VanillaAutoEncoder(_AbstractAutoencoder):
    """Class implementing autoencoder based on feedforward ANN.
    """
    def __init__(self, X_shape, model_tag=None,
                 defined_architecture=None):
        """
        Initialize a VanillaAutoEncoder object, if a pre trained model is not
        loaded an architecture schema has to be passed. In case a pre-trained
        model is loaded this will allow for a warm start.

        Varying the parameters n_layers and corruption, an undercoplete or
        denoising autoencoder can be obtained.

        Args:
            - X_shape: a tuple,
            - model_tag: a string,
            - defined_architecture: is a dictionary defining the components of
                                    the model create_architecture

              The dictionary has the following structure:

                { n_layers :            is a list where its length define the
                                        total number of layers
                                        and the ith element define the number
                                        of hidden units for the ith layer

                  , latent_space :      is an inger specifying the number of
                                        hidden units for the last layer of the
                                        encoder part
                                        this will also define the
                                        dimensionality of the transformed data

                  , activation :        is a string specifying the activation
                                        function for each layer
                                        this is not a flexible option but it
                                        is easier to implement

                  , corruption:         string, specifying the type of
                                        corruption applied
                                        (None, Dropout, Gaussian Noise)

                  , amm_corruption:     float specifying the percentage of
                                        post activation units which are masked
                                        to 0
                                        helps avoiding overfitting

                  , loss :              string or callable specifying the the
                                        loss function employed by the model


                  , output_activation:  is a string specifying the activation
                                        function for the last layer.

                  , optimizer:          string or callable specifying the
                                        optimization algorithm
                }
        """
        if model_tag is None:
            self.model_tag = 'vanilla_autoencoder'
        else:
            self.model_tag = '{}_autoencoder'.format(model_tag)

        if defined_architecture is not None:
            self.n_layers = defined_architecture['n_layers']
            self.latent_space = defined_architecture['latent_space']
            self.activation = defined_architecture['activation']
            self.corruption = defined_architecture['corruption']
            self.ammount_corruption = defined_architecture['amm_corruption']
            self.loss = defined_architecture['loss']
            self.output_activation = defined_architecture['output_activation']
            self.optimizer = defined_architecture['optimizer']
        else:
            self.n_layers = (100, 75, 50)
            self.latent_space = 25
            self.activation = 'relu'
            self.corruption = 'gaussian'
            self.ammount_corruption = 0.2
            self.loss = 'mse'
            self.output_activation = 'sigmoid'
            self.optimizer = 'adam'
        self.build(X_shape)

    def __getstate__(self):
        """Deleting stored encoder and decoder.
        """
        state = dict(self.__dict__)
        del state['_encoder_decoder']
        del state['_encoder']
        return state

    def build(self, X_shape):
        """
        Method for creating the default architecture

        Arguments:
            - X_train: is a numpy array storing the values
                       that the autoencoder will try to reconstruct

        Returns:
            - None
        """
        # corrupt input
        input = Input(shape=(X_shape[1], ))
        if self.corruption == 'dropout':
            corrupted_input = Dropout(self.ammount_corruption)(input)
        elif self.corruption == 'gaussian':
            corrupted_input = GaussianNoise(self.ammount_corruption)(input)
        else:
            corrupted_input = input

        # encoder layers
        for layer, hidden_units in enumerate(self.n_layers):

            if layer == 0:
                encoder = Dense(hidden_units)(corrupted_input)
                encoder = Activation(self.activation)(encoder)
            else:
                encoder = Dense(hidden_units)(encoder)
                encoder = Activation(self.activation)(encoder)

        # latent space
        latent_space = Dense(self.latent_space)(encoder)
        latent_space = Activation(
            self.activation,
            name='latent_space'
        )(latent_space)

        # decoder layers
        for layer, units in enumerate(self.n_layers[::-1]):

            if layer == 0:
                decoder = Dense(hidden_units)(latent_space)
                decoder = Activation(self.activation)(decoder)
            else:
                decoder = Dense(hidden_units)(decoder)
                decoder = Activation(self.activation)(decoder)

        decoder = Dense(X_shape[1])(decoder)
        decoder = Activation(self.output_activation)(decoder)

        # our entire encoder decoder model
        encoder_decoder = Model(input, decoder)
        encoder_decoder.compile(loss=self.loss, optimizer=self.optimizer)
        # we will retain both the entire model
        # as well as a new model created from stacking the imput layer
        # and the encoder part of the autoencoder
        setattr(self, '_encoder_decoder', encoder_decoder)
        setattr(self, '_encoder', Model(input, latent_space))


class RecurrentAutoEncoder(_AbstractAutoencoder):
    """Class implementing autoencoder based on RNN.
    """
    def __init__(self, X_shape, model_tag=None,
                 defined_architecture=None):
        """
        Initialize a RecurrentAutoEncoder object, if a pre trained model is not
        loaded an architecture schema has to be passed. In case a pre-trained
        model is loaded this will allow for a warm start.

        Varying the parameter latent_space and corruption, an undercoplete or
        denoising autoencoder can be obtained.

        Args:
            - save_path:    is a string defining the location where the model
                            will be saved or from where it will be loaded
            - id:           is  a string defining the name of the h5 file in
                            whiche the saved model is stored
            - defined_architecture: is a dictionary defining the components of
                                    the model create_architecture

              The dictionary has the following structure:

                { units :               is an integer, specifying the number of
                                        hidden units for the layers preceeding
                                        and following the latent space layer

                  , latent_space :      is an integer specifying the number of
                                        hidden units for the last layer of the
                                        encoder part
                                        this will also define the
                                        dimensionality of the transformed data

                  , corruption:         string, specifying the type of
                                        corruption applied
                                        (None, Dropout, Gaussian Noise)

                  , amm_corruption:     float specifying the percentage of
                                        post activation units which are masked
                                        to 0
                                        helps avoiding overfitting

                  , loss :              string or callable specifying the the
                                        loss function employed by the model

                  , optimizer:          string or callable specifying the
                                        optimization algorithm
                }
        """
        if model_tag is None:
            self.model_tag = 'recurrent_autoencoder'
        else:
            self.model_tag = '{}_autoencoder'.format(model_tag)

        if defined_architecture is not None:
            self.units = defined_architecture['units']
            self.latent_space = defined_architecture['latent_space']
            self.corruption = defined_architecture['corruption']
            self.ammount_corruption = defined_architecture['amm_corruption']
            self.loss = defined_architecture['loss']
            self.output_activation = defined_architecture['output_activation']
            self.optimizer = defined_architecture['optimizer']
        else:
            self.units = 50
            self.latent_space = 25
            self.corruption = 'dropout'
            self.ammount_corruption = 0.2
            self.loss = 'mse'
            self.output_activation = 'relu'
            self.optimizer = 'adam'
        self.__build(X_shape)

    def __getstate__(self):
        """Deleting stored encoder and decoder.
        """
        state = dict(self.__dict__)
        del state['_encoder_decoder']
        del state['_encoder']
        return state

    def __build(self, X_shape):
        """
        Method for creating the default architecture

        Arguments:
            - X_train: is a numpy array storing the values
                       that the autoencoder will try to reconstruct

        Returns:
            - None
        """
        # corrupt input
        input = Input(shape=(X_shape[1], X_shape[2]))
        if self.corruption == 'dropout':
            corrupted_input = SpatialDropout1D(self.ammount_corruption)(input)
        elif self.corruption == 'gaussian':
            corrupted_input = GaussianNoise(self.ammount_corruption)(input)
        else:
            corrupted_input = input

        # encoder layers
        encoder = LSTM(self.units, return_sequences=True)(corrupted_input)

        # latent space
        encoder = LSTM(self.latent_space, name='latent_space')(encoder)
        latent_space = RepeatVector(X_shape[1])(encoder)

        # decoder layers
        decoder = LSTM(self.units, return_sequences=True)(latent_space)
        decoder = Dense(X_shape[2])(decoder)
        decoder = Activation(self.output_activation)(decoder)

        # our entire encoder decoder model
        encoder_decoder = Model(input, decoder)
        encoder_decoder.compile(loss=self.loss, optimizer=self.optimizer)
        # we will retain both the entire model
        # as well as a new model created from stacking the imput layer
        # and the encoder part of the autoencoder
        setattr(self, '_encoder_decoder', encoder_decoder)
        setattr(self, '_encoder', Model(input, encoder))


class ConvAutoEncoder(_AbstractAutoencoder):
    """Class implementing autoencoder based on CNN.
    """
    def __init__(self, X_shape, model_tag=None,
                 defined_architecture=None):
        """
        Initialize a ConvAutoEncoder object, if a pre trained model is not
        loaded an architecture schema has to be passed. In case a pre-trained
        model is loaded this will allow for a warm start.

        Varying the parameters latent_space and corruption, an undercoplete or
        denoising autoencoder can be obtained.

        Args:
            - model_tag:    is  a string defining the name of the h5 file in
                            whiche the saved model is stored
            - defined_architecture: is a dictionary defining the components of
                                    the model create_architecture

              The dictionary has the following structure:

                { units :               is an integer, specifying the number of
                                        hidden units for the layers preceeding
                                        and following the latent space layer

                  , latent_space :      is an inger specifying the number of
                                        hidden units for the last layer of the
                                        encoder part
                                        this will also define the
                                        dimensionality of the transformed data

                  , corruption:         string, specifying the type of
                                        corruption applied
                                        (None, Dropout, Gaussian Noise)

                  , ammount_corruption: float specifying the percentage of
                                        post activation units which are masked
                                        to 0
                                        helps avoiding overfitting

                  , loss :              string or callable specifying the the
                                        loss function employed by the model

                  , optimizer:          string or callable specifying the
                                        optimization algorithm
                }
        """
        if model_tag is None:
            self.model_tag = 'conv_autoencoder'
        else:
            self.model_tag = '{}_autoencoder'.format(model_tag)

        if defined_architecture is not None:
            self.units = defined_architecture['units']
            self.latent_space = defined_architecture['latent_space']
            self.corruption = defined_architecture['corruption']
            self.ammount_corruption = defined_architecture['amm_corruption']
            self.loss = defined_architecture['loss']
            self.activation = defined_architecture['activation']
            self.output_activation = defined_architecture['output_activation']
            self.optimizer = defined_architecture['optimizer']
        else:
            self.units = 32
            self.latent_space = 8
            self.corruption = 'dropout'
            self.ammount_corruption = 0.0
            self.loss = 'mse'
            self.activation = 'relu'
            self.output_activation = 'relu'
            self.optimizer = 'adam'
        self.__build(X_shape)

    def __getstate__(self):
        """Deleting stored encoder and decoder.
        """
        state = dict(self.__dict__)
        del state['_encoder_decoder']
        del state['_encoder']
        return state

    def __build(self, X_shape):
        """
        Method for creating the default architecture

        Arguments:
            - X_train: is a numpy array storing the values
                       that the autoencoder will try to reconstruct

        Returns:
            - None
        """
        # corrupt input
        input = Input(shape=(X_shape[1], X_shape[2], X_shape[3]))
        if self.corruption == 'dropout':
            corrupted_input = SpatialDropout2D(self.ammount_corruption)(input)
        else:
            corrupted_input = input

        # encoder layers
        # block 1
        encoder = Conv2D(
            self.units,
            (3, 3),
            activation=self.activation,
            padding='same'
        )(corrupted_input)
        encoder = MaxPooling2D(
            (2, 2),
            padding='same'
        )(encoder)
        # block 2
        encoder = Conv2D(
            self.units * 2,
            (3, 3),
            activation=self.activation,
            padding='same'
        )(encoder)
        encoder = MaxPooling2D(
            (2, 2),
            padding='same'
        )(encoder)
        # block 3
        latent_space = Conv2D(
            self.units * 3,
            (3, 3),
            activation=self.activation,
            padding='same',
            name='latent_space'
        )(encoder)

        # decoder layers
        # # block 1
        decoder = Conv2D(
            self.units * 3,
            (3, 3),
            activation=self.activation,
            padding='same'
        )(latent_space)
        decoder = UpSampling2D(
            (2, 2),
        )(decoder)
        # # block 2
        decoder = Conv2D(
            self.units*2,
            (3, 3),
            activation=self.activation,
            padding='same'
        )(decoder)
        decoder = UpSampling2D(
            (2, 2)
        )(decoder)

        # output
        decoder = Conv2D(
            X_shape[3],
            (3, 3),
            activation=self.output_activation,
            padding='same'
        )(decoder)

        # our entire encoder decoder model
        encoder_decoder = Model(input, decoder)
        encoder_decoder.compile(loss=self.loss, optimizer=self.optimizer)
        # we will retain both the entire model
        # as well as a new model created from stacking the imput layer
        # and the encoder part of the autoencoder
        setattr(self, '_encoder_decoder', encoder_decoder)
        setattr(self, '_encoder', Model(input, latent_space))
