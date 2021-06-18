# Extended Improved Deep Embedding Clustering
Extensiond of Improved Deep Embedding Clustering to support recurrent and convolutional autoencoders.

## Motivation
The motivation behind this project is to try and extend the [Deep Embedding for Clustering](https://arxiv.org/pdf/1511.06335.pdf) (DEC) and [Improved Deep Embedding for Clustering](https://www.ijcai.org/proceedings/2017/0243.pdf) (IDEC) algorithms for supporting a larger class of autoencoder (specifically recurrent and convolutional).

Given a dataset, both DEC and IDEC attempts to simultaneously learn a lower dimensional embedding of the data and an optimal partition of the same.
## Features
A series of different autoencoders implementations are available 

* Vanilla Autoencoder (using Multilayer Perceptrons)
* Recurrent Autoencoder
* Convolutional Autoencoder

These can either be pre-trained and plugged into our implementation of IDEC or optimized during the IDEC training stage. Running IDEC with a γ=0 and a pretrained autoencoder will be equivalent to run DEC.

## Quick Benchmark

As a quick benachmark we compared the performcance of IDEC on three differente types of data (using the approapriated type of autoencoder) against an alternative method (Mini-Batch KMeans or DTW-based KMeans). The relsults of the clustering analsysi are reported (along with the ground truth) on a 2D reduction of the original dataset as produced by the UMAP algorithm. Given that we had access to the ground truth, we assessed the performance of each clustering algorithm through the (Rand index adjusted for chance)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html] (ARScore).

### Tabular Data - RNA-Seq Dataset
[This collection of data](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq) is part of the RNA-Seq (HiSeq) PANCAN data set, it is a random extraction of gene expressions of patients having different types of tumor.

<p align="center">   
  <img width="900" height="300"src="https://github.com/vb690/IDEC_extended/blob/main/results/figures/tabular.png">
</p>

### Time Series Data - NATOPS Aircraft Handling Signals Dataset
[This collection of data](http://www.timeseriesclassification.com/description.php?Dataset=NATOPS) report tracking of the gestures found in the Naval Air Training and Operating Procedures Standardization (NATOPS) which standardizes general flight and operating procedures for the US naval aircraft.

<p align="center">   
  <img width="900" height="300"src="https://github.com/vb690/IDEC_extended/blob/main/results/figures/time_series.png">
</p>


### Image Data - Olivetti Dataset
[This collection of data](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset) report ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position.

<p align="center">   
  <img width="900" height="300"src="https://github.com/vb690/IDEC_extended/blob/main/results/figures/images.png">
</p>

## References

### DEC Implementations
* https://github.com/XifengGuo/DEC-keras
* https://github.com/vlukiyanov/pt-dec

### IDEC Implementations 
* https://github.com/XifengGuo/IDEC
* https://github.com/dawnranger/IDEC-pytorch
