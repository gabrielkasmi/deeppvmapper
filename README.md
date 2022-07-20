# "DeepSolar" France

[![DOI](https://zenodo.org/badge/428337790.svg)](https://zenodo.org/badge/latestdoi/428337790)

An automated pipeline for large scale detection of solar arrays in France. Detection is based on an Inception v3 model initially pretrained for DeepSolar Germany and performed on aerial images released by the Institut national de l'information géographique et forestière (IGN). The starting point of this work is the [3D-PV-Locator of Mayer et. al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0306261921016937). We propose a large scale evaluation metric based on the <i> Registre national d'installations (RNI) </i> that enables you to measure the accuracy of the PV detection city-wise.

This repository is the source repository for the following publications: 
- Kasmi, G., et al. (2022). <i> DeepSolar tracker: towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping </i>. [Workshop on Machine Learning for Earth Observation @ECML-PKDD 2022](https://sites.google.com/view/maclean22/people?authuser=0). [http://arxiv.org/abs/2207.07466](http://arxiv.org/abs/2207.07466).


## Overview
## Usage

### Data and models

To replicate the results, you'll need to download the data and the models 

#### Data

The data needed for the pipeline is the following :
- <u> The aerial images </u>, which can be downloaded [here](https://geoservices.ign.fr/bdortho)
- <u> The topological data </u>, which can be downloaded [here](https://geoservices.ign.fr/bdtopo)
- <u> The city data </u>, which can be downloaded here [here](https://www.data.gouv.fr/fr/datasets/decoupage-administratif-communal-francais-issu-d-openstreetmap/)

#### Models

The model weights can be downloaded [here](https://cloud.mines-paristech.fr/index.php/s/qKrZyWCjAoNb43U).

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

Then open the `config.yml` file and specify your own paths to the following directories : `source_images_dir`, `source_iris_dir`, `source_commune_dir`, `source_topo_dir` and `model_dir`. Put the classification and segmentation models in `model_dir` and specify their names as well : `cls_model` and `seg_model`. 

Specify which parts of the pipeline you want to run by setting `True` or `False` for `run_classification`, `run_segmentation` and `run_characterization`.

### Running the pipeline

Before running the pipeline, you'll need to generate the auxiliary files. These files gather information on the buildings, IRIS and communes and are used later on to construct the registry. Generate the auxiliary file by typing `./auxiliary.py --dpt={the number of the departement}`. The department number need to be inputed for the script to work.

Once auxiliary files are computed, the main script can be launched. Type `./main.py --dpt={the number of the departement}` to run the registry on your desired department.

### Evaluating the accuracy

Once evaluation has been completed, you can evaluate the accuracy of the registry against the <i> Registre national d'installations (RNI) </i>. The RNI can be downloaded [here](https://www.data.gouv.fr/fr/datasets/?q=Registre%20national%20des%20installations%20de%20production%20d%27%C3%A9lectricit%C3%A9). Select the year corresponding to the year the orthoimages were released. <b> Please download the RNI under `.json` format </b>. Then type

```python
./evaluate.py --dpt={the number of the departement} --filename={filename} --source_dir={source_dir}
```
to evaluate the registry. In this command, `source_dir` corresponds to the directory where the RNI is located and `filename` corresponds to the name of the RNI file downloaded.


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
