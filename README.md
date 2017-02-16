# Bosch Production Line Performance

Implemented xgboost learning algorithm to [Bosch production line data](https://www.kaggle.com/c/bosch-production-line-performance) to predict if a part would pass or fail within Bosch's production line

## Getting Started

### Install
1. Download data from https://www.kaggle.com/c/bosch-production-line-performance/data
2. Place data in 'input' folder and unzip
3. Output files will be saved in 'output' folder

### Prerequisites

Below are the required python packages:

* Xgboost
* Numpy
* Pandas
* Sklearn
* Matplotlib

### Hardware
I recommend using at a minimum r3.xlarge aws ec2 instance to process and model all the data.I used a r3.8xlarge instance during development and hyperparameter tuning steps for increased processing times.

