Coursera Machine Learning Project
========================================================

Summary
-------
The goal of your project is to predict the manner how some exercises were conducted based on the data in the training dataset. My original plan was to test several methods available in the caret package on the training dataset, however I could conduct only the model fitting with *CART method* ("rpart"). Although I have cleaned the training dataset and removed some variables, the *random forest* method ("rf") was running too long, and the *gradient boosting* method ("gbm") generated an out of memory error.
With investigating the correlation between the variables I may get a  better and feasible model prediction.
With the rpart method I could achive a 54% accuracy which is fairly poor, I expect about 40% pass rate in the test dataset.

Background
----------
Using devices such as *Jawbone Up, Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify *how much* of a particular activity they do, but they rarely quantify *how well* they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har

Getting data
------------
The dataset is presented in comma separated format, it doesn't need any preformatting:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
training <- read.csv("pml-training.csv")
```
A quick review of the dataset's summary highlights that there is a lot of variable with missing or invalid data. The imputing method would be unreliable in case of a so high rate of missing data, so I simply discarded those variable and crated a dataset with the remaining variables.

Cleaning data
-------------

```r
variables <- c("user_name", "new_window", "num_window", "roll_belt", "pitch_belt",
               "yaw_belt", "total_accel_belt", "gyros_belt_x", "gyros_belt_y",
               "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z",
               "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm",
               "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x",
               "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y",
               "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",
               "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",
               "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z",
               "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", 
               "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z",
               "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm",
               "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z",
               "accel_forearm_x", "accel_forearm_y", "accel_forearm_z",
               "magnet_forearm_x",  "magnet_forearm_y", "magnet_forearm_z", "classe")
tr <- training[,variables]
```
Checking for near zero variance in variables show no possibility to remove variables:

```r
nsv <- nearZeroVar(tr, saveMetrics=TRUE)
nsv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## user_name             1.100679    0.03057792   FALSE FALSE
## new_window           47.330049    0.01019264   FALSE  TRUE
## num_window            1.000000    4.37264295   FALSE FALSE
## roll_belt             1.101904    6.77810621   FALSE FALSE
## pitch_belt            1.036082    9.37722964   FALSE FALSE
## yaw_belt              1.058480    9.97349913   FALSE FALSE
## total_accel_belt      1.063160    0.14779329   FALSE FALSE
## gyros_belt_x          1.058651    0.71348486   FALSE FALSE
## gyros_belt_y          1.144000    0.35164611   FALSE FALSE
## gyros_belt_z          1.066214    0.86127816   FALSE FALSE
## accel_belt_x          1.055412    0.83579655   FALSE FALSE
## accel_belt_y          1.113725    0.72877383   FALSE FALSE
## accel_belt_z          1.078767    1.52379982   FALSE FALSE
## magnet_belt_x         1.090141    1.66649679   FALSE FALSE
## magnet_belt_y         1.099688    1.51870350   FALSE FALSE
## magnet_belt_z         1.006369    2.32901845   FALSE FALSE
## roll_arm             52.338462   13.52563449   FALSE FALSE
## pitch_arm            87.256410   15.73234125   FALSE FALSE
## yaw_arm              33.029126   14.65701763   FALSE FALSE
## total_accel_arm       1.024526    0.33635715   FALSE FALSE
## gyros_arm_x           1.015504    3.27693405   FALSE FALSE
## gyros_arm_y           1.454369    1.91621649   FALSE FALSE
## gyros_arm_z           1.110687    1.26388747   FALSE FALSE
## accel_arm_x           1.017341    3.95984099   FALSE FALSE
## accel_arm_y           1.140187    2.73672409   FALSE FALSE
## accel_arm_z           1.128000    4.03628580   FALSE FALSE
## magnet_arm_x          1.000000    6.82397309   FALSE FALSE
## magnet_arm_y          1.056818    4.44399144   FALSE FALSE
## magnet_arm_z          1.036364    6.44684538   FALSE FALSE
## roll_dumbbell         1.022388   84.20650290   FALSE FALSE
## pitch_dumbbell        2.277372   81.74498012   FALSE FALSE
## yaw_dumbbell          1.132231   83.48282540   FALSE FALSE
## total_accel_dumbbell  1.072634    0.21914178   FALSE FALSE
## gyros_dumbbell_x      1.003268    1.22821323   FALSE FALSE
## gyros_dumbbell_y      1.264957    1.41677709   FALSE FALSE
## gyros_dumbbell_z      1.060100    1.04984201   FALSE FALSE
## accel_dumbbell_x      1.018018    2.16593619   FALSE FALSE
## accel_dumbbell_y      1.053061    2.37488533   FALSE FALSE
## accel_dumbbell_z      1.133333    2.08949139   FALSE FALSE
## magnet_dumbbell_x     1.098266    5.74864948   FALSE FALSE
## magnet_dumbbell_y     1.197740    4.30129447   FALSE FALSE
## magnet_dumbbell_z     1.020833    3.44511263   FALSE FALSE
## roll_forearm         11.589286   11.08959331   FALSE FALSE
## pitch_forearm        65.983051   14.85577413   FALSE FALSE
## yaw_forearm          15.322835   10.14677403   FALSE FALSE
## total_accel_forearm   1.128928    0.35674243   FALSE FALSE
## gyros_forearm_x       1.059273    1.51870350   FALSE FALSE
## gyros_forearm_y       1.036554    3.77637346   FALSE FALSE
## gyros_forearm_z       1.122917    1.56457038   FALSE FALSE
## accel_forearm_x       1.126437    4.04647844   FALSE FALSE
## accel_forearm_y       1.059406    5.11160942   FALSE FALSE
## accel_forearm_z       1.006250    2.95586586   FALSE FALSE
## magnet_forearm_x      1.012346    7.76679238   FALSE FALSE
## magnet_forearm_y      1.246914    9.54031189   FALSE FALSE
## magnet_forearm_z      1.000000    8.57710733   FALSE FALSE
## classe                1.469581    0.02548160   FALSE FALSE
```

The fitted model
----------------
The "glm" model is not possible, as the outcome variable has more that 2 levels. So my first choice was the rpart method, with the default cross validation method of the caret package:

```r
fit1 <- train(classe ~ ., data=tr, method="rpart")
```

```
## Loading required package: rpart
```

Expected accuracy
-----------------
The model in sample accuracy is 54%, which is faily poor, the expected accuracy on the test dataset must be lower than this.

```r
fit1
```

```
## CART 
## 
## 19622 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03891896  0.5374427  0.40369906  0.03637426   0.05417782
##   0.05998671  0.3973889  0.17641898  0.05543960   0.09317581
##   0.11515454  0.3203188  0.05425158  0.04004372   0.06248624
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03891896.
```

Further investigation
---------------------
The analysis of the correlation between numeric variables (19 highly correlated variables!) shows a possibility for Principial Component Analysis, but unfortunately I had no time to do that.

```r
numVariables <- c("num_window", "roll_belt", "pitch_belt",
               "yaw_belt", "total_accel_belt", "gyros_belt_x", "gyros_belt_y",
               "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z",
               "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm",
               "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x",
               "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y",
               "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",
               "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",
               "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z",
               "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", 
               "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z",
               "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm",
               "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z",
               "accel_forearm_x", "accel_forearm_y", "accel_forearm_z",
               "magnet_forearm_x",  "magnet_forearm_y", "magnet_forearm_z")
M <- abs(cor(tr[,numVariables]))
diag(M) <- 0
View(M)
which(M>0.8, arr.ind=T)
```

```
##                  row col
## yaw_belt           4   2
## total_accel_belt   5   2
## accel_belt_y      10   2
## accel_belt_z      11   2
## accel_belt_x       9   3
## magnet_belt_x     12   3
## roll_belt          2   4
## roll_belt          2   5
## accel_belt_y      10   5
## accel_belt_z      11   5
## pitch_belt         3   9
## magnet_belt_x     12   9
## roll_belt          2  10
## total_accel_belt   5  10
## accel_belt_z      11  10
## roll_belt          2  11
## total_accel_belt   5  11
## accel_belt_y      10  11
## pitch_belt         3  12
## accel_belt_x       9  12
## gyros_arm_y       20  19
## gyros_arm_x       19  20
## magnet_arm_x      25  22
## accel_arm_x       22  25
## magnet_arm_z      27  26
## magnet_arm_y      26  27
## accel_dumbbell_x  35  29
## accel_dumbbell_z  37  30
## gyros_dumbbell_z  34  32
## gyros_forearm_z   47  32
## gyros_dumbbell_x  32  34
## gyros_forearm_z   47  34
## pitch_dumbbell    29  35
## yaw_dumbbell      30  37
## gyros_forearm_z   47  46
## gyros_dumbbell_x  32  47
## gyros_dumbbell_z  34  47
## gyros_forearm_y   46  47
```
