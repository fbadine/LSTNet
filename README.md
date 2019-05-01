# LSTNet
This repository is a Tensorflow / Keras implementation of __*Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*__ paper https://arxiv.org/pdf/1703.07015.pdf

This implementation has been inspired by the following Pytorch implementation https://github.com/laiguokun/LSTNet

## Installation
Clone this prerequisite repository:
```shell
git clone https://github.com/fbadine/util.git
```

Clone this repository:
```shell
git clone https://github.com/fbadine/LSTNet.git
cd LSTNet
mkdir log/ save/ data/
```

Download the dataset from https://github.com/laiguokun/multivariate-time-series-data and copy the text files into LSTNet/data/

## Usage
### Training
There are 4 different script samples to train, validate and test the model on the different datasets:
- electricity.sh
- exchange_rate.sh
- solar.sh
- traffic.sh

### Predict
In order to predict and plot traffic you will need to run `main.py` as follows (example for the electricity traffic)
```shell
python3.6 main.py --data="data/electricity.txt" --no-train --load="save/electricity/electricity" --predict=all --plot --series-to-plot=0 
```

### Running Options
The following are the parameters that the python script takes along with their description:

| Input&nbsp;Parameters  | Default       | Description |
| :-----------------| :------------------| :-----------|
| --data            |                    |Full Path of the data file. __(REQUIRED)__|
| --normalize       |2                   |Type of data normalisation:<br> - 0: No Normalisation<br> - 1: Normalise all timeseries together<br> - 2: Normalise each timeseries alone|
| --trainpercent    |0.6                 |Percentage of the given data to use for training|
| --validpercent    |0.2                 |Percentage of the given data to use for validation|
| --window          |24 * 7              |Number of time values to consider in each input X|
| --horizon         |12                  |How far is the predicted value Y. It is horizon values away from the last value of X (into the future)|
| --CNNFilters      |100                 |Number of output filters in the CNN layer<br>A value of 0 will remove this layer|
| --CNNKernel       |6                   |CNN filter size that will be (CNNKernel, number of multivariate timeseries)<br>A value of 0 will remove this layer|
| --GRUUnits        |100                 |Number of hidden states in the GRU layer|
| --SkipGRUUnits    |5                   |Number of hidden states in the SkipGRU layer|
| --skip            |24                  |Number of timeslots to skip.<br>A value of 0 will remove this layer|
| --dropout         |0.2                 |Dropout frequency|
| --highway         |24                  |Number of timeslots values to consider for the linear layer (AR layer)|
| --initializer     |glorot_uniform      |The weights initialiser to use|
| --loss            |mean_absolute_error |The loss function to use for optimisation|
| --optimizer       |Adam                |The optimiser to use<br>Accepted values:<br> - SGD<br> - RMSprop<br> - Adam|
| --lr              |0.001               |Learning rate|
| --batchsize       |128                 |Training batchsize|
| --epochs          |100                 |Number of training epochs|
| --tensorboard     |None                |Set to the folder where to put the tensorboard file<br>If set to None => no tensorboard|
| --no-train        |                    |Do not train the model|
| --no-validation   |                    |Do not validate the model|
| --test            |                    |Evaluate the model on the test data|
| --load            |None                |Location and Name of the file to load a pre-trained model from as follows:<br> - Model in filename.json<br> - Weights in filename.h5|
| --save            |None                |Full path of the file to save the model in as follows:<br> - Model in filename.json<br> - Weights in filename.h5<br>This location is also used to save results and history as follows:<br> - Results in filename.txt<br> - History in filename_history.csv if --savehistory is passed|
| --no-saveresults  |                    |Do not save results|
| --savehistory     |                    |Save training / validation history in file as described in parameter --save above|
| --predict         |None                |Predict timeseries using the trained model<br>It takes one of the following values:<br> - trainingdata: predict the training data only<br> - validationdata: predict the validation data only<br> - testingdata: predict the testing data only<br> - all: all of the above<br> - None: none of the above|
| --plot            |                    |Generate plots|
| --series-to-plot  |0                   |Series to plot<br>Format: series,start,end<br> - series: the number of the series you wish to plot<br> - start: start timeslot (default is the start of the timeseries)<br> - end: end timeslot (default is the end of the timeseries)|
| --autocorrelation |None                |Autocorrelation plotting <br>Format: series,start,end<br> - series: the number of random timeseries you wish to plot the autocorrelation for<br> - start: start timeslot (default is the start of the timeseries)<br> - end: end timeslot (default is the end of the timeseries)|
| --save-plot       | None               | Location and name of the file to save the plotted images to<br> - Autocorrelation in filename_autocorrelation.png<br> - Training history in filename_training.png<br> - Prediction in filename_prediction.png|
| --no-log          |                    |Do not create logfiles<br>However error and critical messages will still appear|
| --logfilename     |log/lstnet          |Full path of the logging file|
| --debuglevel      |20                  |Logging debug level|


## Results
The followinng are the results that were obtained:

| Dataset       | Width       | Horizon     | Correlation | RSE         |
| :-------------| :-----------| :-----------| :-----------| :-----------|
| Solar         | 28 hours    | 2 hours     | 0.9548      | 0.3060      |
| Traffic       | 7 days      | 12 hours    | 0.8932      | 0.4089      |
| Electricity   | 7 days      | 24 hours    | 0.8856      | 0.3746      |
| Exchange Rate | 168 days    | 12 days     | 0.9731      | 0.1540      |

## Dataset
As described in the paper the data is composed of 4 publicly available datasets downloadable from https://github.com/laiguokun/multivariate-time-series-data:
- __Traffic:__ A collection of 48 months (2015-2016) hourly data from the California Department of Transportation
- __Solar Energy:__ The solar power production records in 2006, sampled every 10 minutes from 137 PV plants in the state of Alabama
- __Electricity:__ Electricity consumption for 321 clients recorded every 15 minutes from 2012 to 2014
- __Exchange Rate:__ A collection of daily average rates of 8 currencies from 1990 to 2016

## Environment
### Primary environment
The results were obtained on a system with the following versions:
- Python 3.6.8
- Tensorflow 1.11.0
- Keras 2.1.6-tf  

### TensorFlow 2.0 Ready
The model has also been tested on TF 2.0 alpha version:
- Python 3.6.7
- Tensorflow 2.0.0-alpha0
- Keras 2.2.4-tf
