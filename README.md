# DeepSolar tracker: towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping

Accepted for the [Workshop on Machine Learning for Earth Observation @ECML-PKDD 2022](https://sites.google.com/view/maclean22/people?authuser=0). Access the paper [here](http://arxiv.org/abs/2207.07466).

## Overview

### Motivation

Photovoltaic (PV) energy is key to mitigating the current energy crisis. However, distributed PV generation, which amounts to half of the PV energy generation, makes it increasingly difficult for transmission system operators (TSOs) to balance the load and supply and avoid grid congestions. Indeed, in the absence of measurements, estimating the distributed PV generation is tough. In recent years, many remote sensing-based approaches have been proposed to map distributed PV installations. However, to be applicable in industrial settings, one needs to assess the accuracy of the mapping over the whole deployment area. We build on existing work to propose an automated PV registry pipeline. This pipeline automatically generates a dataset recording all distributed PV installations' location, area, installed capacity, and tilt angle. It only requires aerial orthoimagery and topological data, both of which are freely accessible online. In order to assess the accuracy of the registry, we propose an unsupervised method based on the <i> Registre national d'installation </i> (RNI), that centralizes all individual PV systems aggregated at communal level, enabling practitioners to assess the accuracy of the registry and eventually remove outliers. We deploy our model on 9 French <i> départements </i> covering more than 50 000 square kilometers, providing the largest mapping of distributed PV panels with this level of detail to date. We then demonstrate how practitioners can use our unsupervised accuracy assessment method to assess the accuracy of the outputs. In particular, we show how it can easily identify outliers in the detections. Overall, our approach paves the way for a safer integration of deep learning-based pipelines for remote PV mapping. 

### Approach

Our detection is summarized by the following diagram, based on [3D-PV-Locator](https://www.sciencedirect.com/science/article/abs/pii/S0306261921016937):

<p align="center">
<img src="https://github.com/gabrielkasmi/dsfrance/blob/main/figs/flowchart.png" width=700px>
</p>

The installations' characteristics that we extract are the following: surface, tilt, installed capacity. In order to assess the accuracy of the estimation of the installed capacity over the whole deployment area, we introduce a metric based on the <i> registre national d'installations </i> (RNI). This measure consists in reaggregating the installations's installed capacities for each city and to compare the aggregation with the reference value. The main advantage of this approach is that it is unsupervised, fast to compute and available over the whole territory of deployment. This accuracy tracking tool enables practitioner to monitor all the outputs produced by the algorithm, thus improving its accountability.

## Usage

### Data

To replicate the results, you'll need to download the data and the models' weights. This can be downloaded on our [Zenodo repository](https://zenodo.org/record/6862675). 

This repository contains all the necessary data to run the pipeline over a small area of 600 km². If you want to run the pipeline over larger areas, you'll only have to download the corresponding aerial images [here](https://geoservices.ign.fr/bdortho) and the topological data [here](https://geoservices.ign.fr/bdtopo). Also make sure to download the RNI for the correct year, accessible [here](https://www.data.gouv.fr/fr/datasets/?q=Registre%20national%20des%20installations%20de%20production%20d%27%C3%A9lectricit%C3%A9).

### Set-up 

Clone the repository and enter it. 

```python
git clone https://github.com/gabrielkasmi/dsfrance.git
cd dsfrance
```

Then, create the environment :

```python
conda env create --file dsfrance.yml
conda activate dsfrance
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
  doi = {10.48550/ARXIV.2207.07466},
  url = {https://arxiv.org/abs/2207.07466},
  author = {Kasmi, Gabriel and Dubus, Laurent and Blanc, Philippe and Saint-Drenan, Yves-Marie},  
  title = {{DeepSolar tracker: towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping}},
  publisher = {arXiv},
  year = {2022},
}

```

Like this work ? Do not hesitate to <a class="github-button" href="https://github.com/gabrielkasmi/dsfrance" data-icon="octicon-star" aria-label="Star gabrielkasmi/dsfrance on GitHub">star</a> us ! 
