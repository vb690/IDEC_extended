from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K


class ClusteringLayer(Layer):
    """Clustering layer converts input sample (feature) to soft label, i.e. a
    vector that represents the probability of the sample belonging to each
    cluster. The probability is calculated with student's t-distribution.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        """Instatiate a clustering layer.
        Args:
            n_clusters: integer number of clusters.
            weights:    numpy array with shape `(n_clusters, n_features)`.
                        witch representing initial cluster centers.
            alpha:      integer specifying  dof Student's t-distribution.

        Returns:
            clusters: 2D tensor with shape: (n_samples, n_clusters), soft.
                cluster assignment
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(
            dtype=K.floatx(),
            shape=(None, input_dim)
        )
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters_centroids'
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        return None

    def call(self, inputs, **kwargs):
        """Compute student t-distribution, as same as used in t-SNE algorithm,
        which is given by

        q_ij = 1/(1+dist(x_i, u_j)^2)

        The results are then normlized.

        Args:
            inputs: tensor containing the data, shape=(n_samples, n_features)

        Returns:
            q:  student's t-distribution, or soft cluster assignment for each
                sample, shape=(n_samples, n_clusters).
        """
        # we compute the distance between the centroids and each point
        euclidian_distance = K.sum(
            K.square(K.expand_dims(inputs, axis=1) - self.clusters),
            axis=2
        )
        q = 1.0 / (1.0 + euclidian_distance / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
