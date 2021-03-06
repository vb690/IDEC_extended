
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt


def compare_clusterers(clusterer_1, clusterer_2, ground_truth, reduction,
                       figsize=(15, 5), **kwargs):
    """Utility function for comparing two clustering approaches on a problem
    for which we have a ground truth.

        Args:
            - clusterer_1: a dictionary where key is the name of the clustering
                method and value are the predicted labels.
            - clusterer_2: a dictionary where key is the name of the clustering
                method and value are the predicted labels.
            - ground_truth: a numpy array, the ground truth labels.
            - reduction: a numpy array, a matrix holding the low dimensional
                projection of the original input data.
            - figsize: a tuple, specifying the size of the figure
            - **kwargs: keyword arguments passed to plt.scatter.
        Returns:
            - None
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    score_1 = adjusted_rand_score(
        labels_true=ground_truth,
        labels_pred=next(iter(clusterer_1.values()))
    )
    score_2 = adjusted_rand_score(
        labels_true=ground_truth,
        labels_pred=next(iter(clusterer_2.values()))
    )

    axs[0].scatter(
        reduction[:, 0],
        reduction[:, 1],
        c=ground_truth,
        **kwargs
    )
    axs[0].set_title('Ground Truth')
    axs[1].scatter(
        reduction[:, 0],
        reduction[:, 1],
        c=next(iter(clusterer_1.values())),
        **kwargs
    )
    axs[1].set_title(
        f'Clust {list(clusterer_1.keys())[0]} - ARScore {round(score_1, 2)}'
    )
    axs[2].scatter(
        reduction[:, 0],
        reduction[:, 1],
        c=next(iter(clusterer_2.values())),
        **kwargs
    )
    axs[2].set_title(
        f'Clust {list(clusterer_2.keys())[0]} - ARScore {round(score_2, 2)}'
    )

    plt.show()
    return None
