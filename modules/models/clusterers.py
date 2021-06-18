import numpy as np

from sklearn.cluster import MiniBatchKMeans

from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from ..utils.abstract_models import _AbstractDEC
from ..utils.custom_classes import ClusteringLayer


class IDEC(_AbstractDEC):
    """Class implementing an improved version of DEC from:

        * This paper: https://www.ijcai.org/proceedings/2017/0243.pdf

    The model attempts to simultaneously learn a feature representation
    and cluster assignement similarly to DEC. Howevre, the model jointly
    optimize the reconstruction and clustering loss differently from DEc which
    only optimize for the latter, in this way the new learned features are
    still forced to be a 'good approximation' for recontructing the original
    input.
    """
    def __init__(self, X, autoencoder, n_clusters, gamma=0.1, model_tag=None,
                 initial_centroids=None, labels=None, optimizer=Adam()):
        """
        Method called when instatiating the VanillaDEC class

        Args:
            - X:            a numpy array, input features that will be
                            clustered
            - autoencoder:  an autoencoder obejct, it needs to be of the type
                            implemented in the autoencoders module
            - n_clusters:   an integer, specifying the number of expected
                            clusters
            - gamma:        a float, sclaing factor controlling how much the
                            clustering layer will distort the embedding space
                            (gamma=0 is a simple autoencoder + k means)
            - initial_centroids:    a list of lists specifying the coordinates
                                    in the feature space of each centroid
            - optimizer: a string specifying the keras optimizer to employ
        """
        if model_tag is None:
            self.model_tag = 'IDEC'
        else:
            self.model_tag = '{}_IDEC'.format(model_tag)

        self.X = X
        self.n_clusters = n_clusters
        if initial_centroids is not None:
            self.initial_centroids = [initial_centroids]
        else:
            self.initial_centroids = None
        self.labels = labels
        self.optimizer = optimizer
        self.gamma = gamma
        self.autoencoder_loss = autoencoder._encoder_decoder.loss

        # we build the model when instatiating the class
        self.build(
            X=X,
            autoencoder=autoencoder._encoder_decoder,
            encoder=autoencoder._encoder,
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_model']
        return state

    def build(self, X, autoencoder, encoder):
        """Method for building the model

        Arguments:
            - X: is a numpy array, input features that will be clustered
            - autoencoder: a keras model, autoencoder used for getting the
              embedding.
            - encoder: a keras model, encoder portion of the autoencoder.

        Returns
            - None
        """
        input_autoencoder = autoencoder.get_layer('latent_space').output
        input_autoencoder = Flatten()(input_autoencoder)

        clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            weights=self.initial_centroids,
            name='clustering'
        )(input_autoencoder)

        model = Model(
            inputs=autoencoder.input,
            outputs=[clustering_layer, autoencoder.output]
        )

        model.compile(
            optimizer=self.optimizer,
            loss=['kld', autoencoder.loss],
            loss_weights=[self.gamma, 1.0]
        )
        if self.initial_centroids is None:
            clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                n_init=50,
                max_iter=1000,
                batch_size=256
            )
            embedding = encoder.predict(
                X,
                batch_size=256,
                verbose=0
            )
            embedding = embedding.reshape(embedding.shape[0], -1)
            clusterer.fit(embedding)
            model.get_layer('clustering').set_weights(
                [clusterer.cluster_centers_]
            )
        setattr(self, '_model', model)

    def fit(self, X, epochs, btch_size, update_interval=140, verbose=True,
            tol=1e-3):
        """Method for fitting the IDEC to the data

        Arguments:
            - X: a numpy array, input data to cluster
            - epochs: an integer, number of epochs (approximation) for which
              the algorithm is run.
            - batch_size: an integer, size of the batch of data passed at
              each step.
            - update_interval: an integer specifying after how many batches to
              produce a target distribution for self learning.
            - verbose: is a bolean, specifying if training status is printed.
            - tol: is a float, specifying the mimimum decrease in the clusters
              movements for continuing training.

        Returns:
            - None
        """
        # set the first prediction for monitoring convergence
        if self.labels is None:
            if len(X.shape) > 2:
                labels_X = X.reshape(X.shape[0], -1)
            else:
                labels_X = X
            temp_pred = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                n_init=50,
                max_iter=1000,
                batch_size=256
            ).fit_predict(labels_X)
        else:
            temp_pred = self.labels
        temp_pred_last = np.copy(temp_pred)

        maxiter = (X.shape[0] // btch_size) * epochs
        status = {
            'epoch': 0,
            'loss': 0,
            'kl_divergence': 0,
            self.autoencoder_loss: 0,
        }

        # start batch training
        btch_index = 0
        for ite in range(int(maxiter)):

            if ite % update_interval == 0:
                prediction = self._model.predict(
                    X,
                    batch_size=btch_size,
                    verbose=0
                )
                q = prediction[0]
                # update the auxiliary target distribution p
                p = self._generate_target_distribution(q)

                # check convergence
                temp_pred = q.argmax(1)
                delta = np.sum(
                    temp_pred != temp_pred_last
                ).astype(np.float32) / temp_pred.shape[0]
                temp_pred_last = np.copy(temp_pred)
                if ite > 0 and delta < tol:
                    print('Tol: {} Delta: {}'.format(tol, delta))
                    print('Stop Training')
                    break

            # train on batch
            if (btch_index + 1) * btch_size > X.shape[0]:
                loss = self._model.train_on_batch(
                    x=X[btch_index * btch_size::],
                    y=[
                        p[btch_index * btch_size::],
                        X[btch_index * btch_size::]
                      ]
                )
                status['loss'] = loss[0]
                status['kl_divergence'] = loss[1]
                status[self.autoencoder_loss] = loss[2]
                status['epoch'] += 1
                if verbose:
                    self._print_status(status)
                btch_index = 0
            else:
                loss = self._model.train_on_batch(
                    x=X[btch_index * btch_size:(btch_index + 1) * btch_size],
                    y=[
                        p[btch_index * btch_size:(btch_index + 1) * btch_size],
                        X[btch_index * btch_size:(btch_index + 1) * btch_size]
                    ]
                )
                status['loss'] = loss
                btch_index += 1

    def predict(self, X, **kwargs):
        predictions = self._model.predict(X, **kwargs)
        soft_labels = predictions[0]
        return soft_labels
