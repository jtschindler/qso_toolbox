# The "QSO Toolbox" package

---
---


This package contains modules with high level functions for quasar selection.

## Installation via Anaconda (recommended)

To install the QSO Toolbox follow these steps:

* Clone the repository: `git clone https://github.com/jtschindler/qso_sel_toolbox.git`
* Use the environment.yml file to set up a install a new conda environment: `conda env create --file environment.yml`
* Download the NOIR Datalab github repo `git clone https://github.com/noaodatalab/datalab.git`.
* Install the NOIR datalab package `python setup.py install`
* Go back to the qso_sel_toolbox folder and install the package with `python setup.py install`

## Manual installation (not recommended)
* Clone the repository: `git clone https://github.com/jtschindler/qso_sel_toolbox.git`
* Check the 'environment.yml' file for dependencies and install them via pip or anaconda
* Download the NOIR Datalab github repo `git clone https://github.com/noaodatalab/datalab.git`.
* Install the NOIR datalab package `python setup.py install`
* Go back to the qso_sel_toolbox folder and install the package with `python setup.py install`

## Getting started and examples

In the `qso_sel_toolbox/examples` directory you will find ipython notebooks, which guide you through some of
 the basic functionality og this package. I recommend to make use of these materials.  

---

## Modules

### Service modules

In the following we introduce the separate modules and their main capabilities.

#### utils - general utility module
This module hosts more general functions that can be used by all the other modules. Currently these functions include conversion of coordinates from decimal degrees to HMS/DMS format.


#### photometry_tools - module for manipulating photometry
This modules contains a range of functions for manipulating photometry, like flux conversions, dereddening, and Vega to AB conversions.

---
### Main modules

#### catalog_tools
This module includes functions that query and retrieve astronomical catalog data from on-line sources.

### Current functionality:
#### * Retrieving offset stars (2MASS, Nomad, VHSDR6, DES DR1, PS1)
#### * Download of catalog images or image cutouts (PS1, VHSDR6, DES DR1, unWISE)
#### * Two helper functions to manually download images from the VISTA Science Archive (deprecated)

## image_tools
This module focuses on forced photometry calculation and high-level plotting routines.

### Current functionality:
#### * Plotting multi-survey image data with additional photometry information
#### * Create finding charts (utilizing the output of the offset star retrieval)
#### * Forced photometry calculation (unWISE, DES DR1)
