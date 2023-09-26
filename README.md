# DeepPVMapper

## Overview

DeepPVMapper is a deep learning-based mapping algorithm developped to map rooftop PV installations over France. This algorithm is still under development and the repository hosts the latest release of the algorithm.

The latest version of the Zenodo repository is accessible here : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7576814.svg)](https://doi.org/10.5281/zenodo.7576814)


This algorithm is used in the following papers:
* "Towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping", accepted for the [Workshop on Machine Learning for Earth Observation @ECML-PKDD 2022](https://sites.google.com/view/maclean22/people?authuser=0).
  * Access the paper here : [https://arxiv.org/abs/2207.07466](https://arxiv.org/abs/2207.07466)
  * Stable code corresponding to the paper : [https://github.com/gabrielkasmi/deeppvmapper/tree/workshop](https://github.com/gabrielkasmi/deeppvmapper/tree/workshop)
  * Stable repository corresponding to the paper : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6862675.svg)](https://doi.org/10.5281/zenodo.6862675)

* "DeepPVMapper: reliable and scalable remote sensing of rooftop photovoltaic installations"
  * Stable code corresponding to the paper: [![DOI](https://zenodo.org/badge/428337790.svg)](https://zenodo.org/badge/latestdoi/428337790)
  * Stable repository corresponding to the paper: [https://github.com/gabrielkasmi/deeppvmapper/tree/pscc](https://github.com/gabrielkasmi/deeppvmapper/tree/pscc)

## Approach

Our detection is summarized by the following diagram, based on [3D-PV-Locator](https://www.sciencedirect.com/science/article/abs/pii/S0306261921016937):

<p align="center">
<img src="https://github.com/gabrielkasmi/dsfrance/blob/main/figs/flowchart.png" width=700px>
</p>

The installations' characteristics that we extract are the following: surface, tilt, installed capacity. In order to assess the accuracy of the estimation of the installed capacity over the whole deployment area, we introduce a metric based on the <i> registre national d'installations </i> (RNI). This measure consists in reaggregating the installations's installed capacities for each city and to compare the aggregation with the reference value. The main advantage of this approach is that it is unsupervised, fast to compute and available over the whole territory of deployment. This accuracy tracking tool enables practitioner to monitor all the outputs produced by the algorithm, thus improving its accountability. We trained our classification and segmentation models on a new training database called BDAPPV, which you can access [here](https://www.nature.com/articles/s41597-023-01951-4).

## Usage

### Data

To replicate the results, you'll need to download the data and the models' weights. This can be downloaded on our Zenodo repository (see the latest release above)

This repository contains all the necessary data to run the pipeline over a small area of 120 km². If you want to run the pipeline over larger areas, you'll only have to download the corresponding aerial images [here](https://geoservices.ign.fr/bdortho) and the topological data [here](https://geoservices.ign.fr/bdtopo). Also make sure to download the RNI for the correct year, accessible [here](https://www.data.gouv.fr/fr/datasets/?q=Registre%20national%20des%20installations%20de%20production%20d%27%C3%A9lectricit%C3%A9).

### Set-up 

Clone the repository and enter it. 

```python
git clone https://github.com/gabrielkasmi/deeppvmapper.git
cd deeppvmapper
```

Then, create the environment :

```python
conda env create --file deeppvmapper.yml
conda activate deeppvmapper
```

### Replication

We recommend that you follow our `hands-on.ipynb` notebook, located in the folder `notebooks`. This notebook will present you how to set up the configuration file, run the initialization script, the main pipeline and the evaluation script. This notebook maps an area of approximately 600 km². You can also directly run the scripts from the terminal, without going through the notebook.

### Evaluating the accuracy

To evaluate the accuracy, you can execute the script `evaluate.py`. In addition, you can go through the `visualization.ipynb` notebook, located in the folder `notebooks`. In this notebook, you will explore the registry that you will have generated in the `hands-on.ipynb` notebook.

## License and citation

### License

This software is provided under [GPL-3.0 license](https://github.com/gabrielkasmi/dsfrance/blob/main/LICENSE). 

### Citation: 

```
@article{kasmi2022deepsolar,
  title={Towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping},
  author={Kasmi, Gabriel and Dubus, Laurent and Blanc, Philippe and Saint-Drenan, Yves-Marie},
  journal={arXiv preprint arXiv:2207.07466},
  year={2022}
}
```

Like this work ? Do not hesitate to <a class="github-button" href="https://github.com/gabrielkasmi/deeppvmapper" data-icon="octicon-star" aria-label="Star gabrielkasmi/deeppvmapper on GitHub">star</a> us ! 
