# `pyexspec`
An automated pipeline for reducing 2D spectra of instruments and is a modified version of [bfosc](https://github.com/hypergravity/bfosc/)

Authors:
- Jiao Li
- Zhang Bo

 

## Installation

1. install `songcn` package
    - `pip install -U git+git://github.com/lidihei/songcn`
    - `pip show songcn` should be at least `0.0.9`
    - or install 'songcn' package by
    - `git clone https://github.com/lidihei/songcn.git`
    - `bash install.sh`
2. install `pyexspec` package
    - `git clone https://github.com/lidihei/pyexspec.git`
    - `cd pyexspec`
    - `pip install .`

## Extracting spectrum of E9G10 of Xinglong 216cm
    - cd pyexspec/bfoscE9G10/gui
    - python app.py
- - if extract BFOSC the rot90 must be selected
<p align="center"><img width="100%" src="bfoscE9G10/bfosc_gui.png" /></p>

- - bfosc orders reference
<p align="center"><img width="100%" src="bfoscE9G10/template/bfoscE9G10_orders_rot90.png" /></p>
   

## yfosc E9G10
<p align="center"><img width="100%" src="yfoscE9G10/template/yfosc_fear.png" /></p>


## Manually callibrate wavelength
-  `cd wvcalib`
- `$python wvclib_app.py`
- - when set value of emission line, you shoud press "Enter/Return" keyboard after input the value into the table.
- - Finding the emission line automatically:
- - - BFOSC E9G10
- - - - npix_chunck = 8; CCF_Kernel Width = 1.5; num_sigma_clip = 3
- - - - Fitting Function: Poly2DFitter; Parameter : deg(X) = 4; deg(Y) = 6 
- - - YFOSC E9G10
- - - - npix_chunck = 5; CCF_Kernel Width = 1.5; num_sigma_clip = 3
- - - - Fitting Function: Poly2DFitter; Parameter : deg(X) = 4; deg(Y) = 6 


