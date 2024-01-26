# LoTSS YOLO Detection

## Introduction

This project takes place in an international effort to develop new data processing and analysis tools for the upcoming Square Kilometer Array (SKA), the first light of which is scheduled for 2028.
This instrument will reach an unprecedented sensitivity, allowing it to set new constraints on the early stages of the universe and better constrain the evolution of astronomical objects over cosmological times. However, the analysis of the data produced by the SKA will be very challenging. The current forecasts predict up to 700 PB of archived data per year and a raw data output of around 1 TB per second, which must be processed in real time to produce science data products. With such a data rate and with data products that are large and highly dimensional, current widely adopted analysis methods exhibit significant limitations. For this reason, new innovative analysis methods must be developed alongside the SKA deployment. For this, the astronomical community makes use of current instruments that are considered "pathfinders" or "precursors" for the SKA (e.g., ASKAP, MeerKAT, LOFAR, etc.).

In this context, the SKAO (SKA Observatory) started a series of data challenges (SDC), intending to provide simulated data products that should be representative of the SKA.
These challenges aim to compare analysis tools on controlled datasets and encourage the development of new data analysis methods.
While these SDCs seek to represent a variety of analysis tasks, the first two editions of the challenges (SDC1 Bonaldi et al. (2020) and SDC2 Hartley et al. (2023)) focused on source detection and characterization.
The first edition was about detecting sources in simulated 2D continuum images, and the second edition in a simulated 3D HI emission cube.

To participate in the SDC2, members of the MINERVA project from the Paris Observatory developed a deep-learning approach for source detection and characterization that would be applied to both 2D and 3D datasets.
This new innovative method demonstrated state-of-the-art performance on both types of simulated data.
The team noticeably reached first place in the SDC2 and achieved much better results than all the solutions submitted during the SDC1.
The method must now be generalized on observational radio datasets, which can be done using various surveys from precursor instruments.

This project lies in this global effort of applying the method to existing radio datasets.
I noticeably focused on trying to apply the detector trained on the first SDC on observational continuum surveys, namely the LoFAR Two-metre Sky Survey (LoTSS DR2 Shimwell, et al. (2022)).
This survey is already associated with a source catalog derived using the same classical detection method.
This catalog will be considered as a reference to which our detection results will be systematically compared.
Because MINERVA's source detector is based on supervised machine learning, it must be trained for the survey it will be applied to.
This can be done in two different ways: i) by using simulated examples, or ii) by using actual observations with labels from another method or confirmed observationally using other data.

This project is separated into two steps.
The first step is to evaluate MINERVA's method capabilities on the selected survey by performing source detection using the network trained on the simulated data from the SDC1.
By doing this, we will evaluate if the SDC1 is a good enough approximation of observational data for the LoTSS.
The second step consists of using observational examples to perform complementary network training to better account for the survey's specificities.
This approach requires the construction of a comprehensive and high-quality source catalog based on a different method than the one used to define the reference source catalog.

## Content

You can find the details concerning the environment in the [wiki](https://github.com/ADnothing/LoTSS-YOLO-Detection/wiki/Environment).

### [Trainingset_utils](ttps://github.com/ADnothing/LoTSS-YOLO-Detection/wiki/Environment](https://github.com/ADnothing/LoTSS-YOLO-Detection/wiki/Trainingset_utils)

This folder contains the function used to build the training dataset to perform the fine training of our neural network.

| File Name            | Description                                                   |
| -------------------- | --------------------------------------------------------------|
| make_cat.py          | Contains the functions to build the initial catalogs.         |
| corr_cat.py          | Contains the functions to clean the produced catalogs         |
| CrossMatch.py        | Contains the function to perform cross-matches                |
| main.py              | Main python file                                              |
| config.ini           | Example of the config file (used as a parameter of main.py)   |
| config_LoTSS.ini     | Config file used for LoTSS                                    |
