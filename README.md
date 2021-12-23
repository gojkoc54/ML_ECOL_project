## Automatic detection of natural slicks in Lake Geneva from a ground-based optical imagery package
**(EPFL ML4Science project, Meachine Learning CS-433)**

Slicks are smooth surface areas that often appear in lakes due to various reasons. Ecological Engineering Laboratory (ECOL) acquires an extensive database of RGB images taken by a ground-based imagery system installed on the northern shore of Lake Geneva. An essential task for further research into the slick formation, morphology, and kinematics is their automated detection in a large corpus of taken images. We created a dataset of cropped, resized, and clean images from the raw data. We propose two approaches for solving this problem: image processing with basic machine learning models and a CNN-based image classifier. *Random Forest Classifier* achieved a categorical accuracy of 80.14\%, while the *Densenet121* architecture reached a categorical accuracy of 91.73\% on the test set. 


## Structure of the project

```
ML_ECOL_project/
    |
    |-- src/
    |   |-- data_cleaning.ipynb
    |   |-- basic_ml_approach.ipynb
    |   |-- cnn_results_visualization.ipynb
    |   |
    |   |-- train_cnn_model.py
    |   |-- eval_cnn_model.py
    |   |-- training_utils.py
    |   |-- utils.py
    |   |
    |   |-- run_train.sh
    |   |-- run_eval.sh
    |
    |-- report.pdf
    |
    |-- README.md
```

* ```data_cleaning.ipynb``` - notebook containing initial data cleaning code (cropping, converting, compressing, ... )
* ```cnn_results_visualization.ipynb``` - notebook containing visualizations of train/val and test results of our CNN models
* ```basic_ml_approach.ipynb``` - notebook containing our Image Processing + ML approach
* ```train_cnn_model.py``` - script that trains a given model (loads its pretrained version, replaces the classification layer and performs fine-tuning)
* ```eval_cnn_model.py``` - script that evaluates a trained model (loads a checkpoint made by the training script and evaluates it on the test set)
* ```training_utils.py``` - file containing functions used for training the models (train_epoch, validation_epoch, fit_model, ... )
* ```utils.py``` - file containing functions that are used all over the project (dataset definition, data loading, ... )
* ```run_train.sh``` - script that runs the training script for all models
* ```run_eval.sh``` - script that runs the evaluation script for all models
* ```report.pdf``` - the final Latex report of our project


## Team info

* Team name: **ISAAA**
* Team members:
    * Gojko Cutura - gojko.cutura@epfl.ch
    * Vuk Vukovic - vuk.vukovic@epfl.ch
    * Radenko Pejic - radenko.pejic@epfl.ch


