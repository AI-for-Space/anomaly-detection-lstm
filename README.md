# Detección de anomalía no supervisada con líneas temporales univariadas aplicadas a datos satelitales

<div align=center>
  <a href="https://www.universidadviu.com/es/"><img src="https://user-images.githubusercontent.com/15159632/155946766-9bf49086-a07f-473c-a703-65c1cc739c9c.png" alt="VIU" title="VIU" hspace="30" height="96px" /></a>
</div>

<br/>

Author: Jorge Vergara ([@jorge1993](https://github.com/jorge1993)) (jorgevergara1993@gmail.com)

Advisor: Guzman-Alvarez, Cesar Augusto ([@cguz](https://github.com/cguz)) 

---

## Objective

Perform anomaly detection using the Online evolving Spiking Neural Networks for Unsupervised Anomaly Detection (OeSNN-UAD) technique, which has an unsupervised approach based on univariate timelines

## Input Data Format

Each input data file should contain three columns: the first one being series of timestamps, the second one being real input values to be classified and the third one should contain labels denoting presence or absence of anomaly for each input value. The presence of anomaly should be indicated by 1, while absence by 0. The types of these three columns should be as follows:
  * timestamp: string,
  * input value: real (double, float),
  * anomaly label: Boolean (1 or 0) *Optional*

The input data file should contain column headers, even if his name doesn't matter. Rows in data files should not be numbered. The separator at the .csv file must be semicolon ';'.

The input data file path, have to be absolute.

To replicate the output data, all the input files are on *Data/*.

## Output Data Format

On each executions all the results are on the folder *Data/Results*. Contains two files, one for the *results.csv* whitch contains the raw data with this columns: 
* X value
* Y value
* labels

The other file is a representive graph of the results.

## Paper
Unsupervised Anomaly Detection in Stream Data with Online Evolving Spiking Neural Networks - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/The-proposed-OeSNN-UAD-architecture_fig2_349596550 [accessed 17 Oct, 2022]

