# Electricity Consumption Surveys

Estimating the demand for electricity is a critical step in the design of a medium to long term energy plan. Frequently, estimates are based on time series with few observation points or on data from other countries or regions. Microsimulation models use household surveys to offer an alternative estimation route based on observed electricity demand by households with different incomes.

## Getting Started

This repository contains a python module : [IPCmicrosimTool](https://github.com/UN-DESA-Modelling/Electricity_Consumption_Surveys/blob/master/ipc_microsim_tool/ipc_microsim_tool.py) that can be used to run micro-simulation. It contains 2 basic component : 
* IPCmicrosimTool module
* [iPython](https://github.com/UN-DESA-Modelling/Electricity_Consumption_Surveys/blob/master/ipc_microsim_tool/ipc_microsim_tool.ipynb) file for example on using the module, which are using :
* Astlan (pseudo country) data
* World data

### Prerequisities

* Python 2.7 installed on your machine. Follow the installation instruction on their downloads [page](https://www.python.org/downloads/). If you are using [brew](http://brew.sh/) on your OSX, run this command on your terminal : 
```
brew install python
```
* Required libraries installed on your machine. Run this command on your terminal : 
```
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install patsy
pip install statsmodel
```
* (Optional) iPython installed on your machine
```
pip install ipython
pip install mpltools
```

### Installing
1. Clone this repository / download the zip file
2. Navigate to ipc_microsim_tool folder

#### Method 1 : Using python CLI
3. Run python by typing this command on your terminal
```
python
```
4. Import the module
```
from ipc_microsim_tool import IPCmicrosimTool as imt
```
#### Method 2 : Using ipython
5. Run ipython by typing this command on your terminal
```
ipython notebook
```
6. Navigate to localhost:8888/tree on your browser
7. Open ipc_microsim_tool.ipynb


## Built With

* python (2.7)
* numpy
* scipy
* pandas
* matplotlib
* patsy
* statsmodels

## Authors

* **Rafael Guerreiro Osorio** - *Initial work* - [Instituto de Pesquisa Econômica Aplicada](www.ipea.gov.br) - [International Policy Centre for Inclusive Growth](www.ipc-undp.org)
