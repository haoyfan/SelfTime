# DodgersLoop data sets

The traffic data are collected with the loop sensor installed on ramp for the 101 North freeway in Los Angeles. This location is close to Dodgers Stadium; therefore the traffic is affected by volumne of visitors to the stadium. 

We make three data sets out of these data.

## DodgersLoopDay

The classes are days of the week.

- Class 1: Sunday
- Class 2: Monday
- Class 3: Tuesday
- Class 4: Wednesday
- Class 5: Thursday 
- Class 6: Friday
- Class 7: Saturday 

Train size: 78

Test size: 80

Number of classes: 7

Missing value: Yes

Time series length: 288

Missing values are represented with NaN.

## DodgersLoopWeekend

- Class 1: Weekday
- Class 2: Weekend

Train size: 20

Test size: 138

Number of classes: 2

Missing value: Yes

Time series length: 288

Missing values are represented with NaN.

## DodgersLoopGame

- Class 1: Normal day
- Class 2: Game day

Train size: 20

Test size: 138

Number of classes: 2

Missing value: Yes

Time series length: 288

There is nothing to infer from the order of examples in the train and test set.

Missing values are represented with NaN.

Data created by Ihler, Alexander, Jon Hutchins, and Padhraic Smyth (see [1][2][3]). Data edited by Chin-Chia Michael Yeh. 

[1] Ihler, Alexander, Jon Hutchins, and Padhraic Smyth. "Adaptive event detection with time-varying poisson processes." Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2006.

[2] “UCI Machine Learning Repository: Dodgers Loop Sensor Data Set.” UCI Machine Learning Repository, archive.ics.uci.edu/ml/datasets/dodgers+loop+sensor.

[3] “Caltrans PeMS.” Caltrans, pems.dot.ca.gov/.
