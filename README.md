# "DeepSolar" France

An automated pipeline for large scale detection of solar arrays in France. Detection is based on an Inception v3 model initially pretrained for DeepSolar Germany and performed on aerial images released by the Institut national de l'information géographique et forestière (IGN). The starting point of this work is the [3D-PV-Locator of Mayer et. al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0306261921016937).

## Overview
## Usage

### Data and models

To replicate the results, you'll need to download the data and the models 

#### Data

The data needed for the pipeline is the following :
- <u> The aerial images </u>, which can be downloaded [here](https://geoservices.ign.fr/bdortho)
- <u> The topological data </u>, which can be downloaded [here](https://geoservices.ign.fr/bdtopo)
- <u> The IRIS and Commune data </u>, which can be downloaded [here](https://geoservices.ign.fr/contoursiris) and [here](https://www.data.gouv.fr/fr/datasets/decoupage-administratif-communal-francais-issu-d-openstreetmap/)
- <u> The surface models </u>

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


## License and citation
This software is provided under [GPL-3.0 license.](https://github.com/gabrielkasmi/dsfrance/blob/main/LICENSE).
