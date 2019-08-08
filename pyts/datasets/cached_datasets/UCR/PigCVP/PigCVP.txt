# InternalBleeding dataset

The original data are from 52 pigs having three vital signs monitored before and after an induced injury. 

We make three datasets out of this data.

Class is the individual pig. In the training set, a class is represented by two examples: the first 2000 data points of the *before* time series and the first 2000 data points of the *after* time series. 

In the test set, a class is represented by four examples, the second and third 2000 data points of the *before* time series and the second and third 2000 data points of the *after* time series. 

## PigAirwayPressure
Airway pressure measurements

Train size: 104

Test size: 208

Missing value: No

Number of classses: 52

Time series length: 2000

## PigArtPressure
Arterial blood pressure measurements

Train size: 104

Test size: 208

Missing value: No

Number of classses: 52

Time series length: 2000

## PigCVP
Central venous pressure measurements

Train size: 104

Test size: 208

Missing value: No

Number of classses: 52

Time series length: 2000

There is nothing to infer from the order of examples in the train and test set.

Data created by Mathie Guillame-Bert et al. (see [1]). Data edited by Shaghayegh Gharghabi and Eamonn Keogh.

[1] Guillame-Bert, Mathieu, and Artur Dubrawski. "Classification of time sequences using graphs of temporal constraints." The Journal of Machine Learning Research 18.1 (2017): 4370-4403.
