# OeSNN-UAD

## Source code files



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



## Contact

For questions contact Jorge Vergara: <jorgevergara1993@gmail.com>