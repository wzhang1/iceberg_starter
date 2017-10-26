# iceberg_starter
mxnet starter for https://www.kaggle.com/c/statoil-iceberg-classifier-challenge
modified from https://github.com/zackchase/mxnet-the-straight-dope

### Data preparation

download train.json and test.json to the folder input, run 
'''
./creat_files.sh
'''
to create idx files, a 80% train split is used, need to be improved.

### Train

I tried resnet18 as an example, in script folder

'''
python gluon.py
'''

### Predict
Predict script need to be modified to have the stored params from traning
'''

'''

## To do

1, It seem that some prediction is 0 or 1 which make leadbboard score bad,
need to find a better way to cap the predicted values. Using this script as is
can only get 0.61 in public lb. A larger resolution is used which may be a bad idea.
Data generation is navie, may affect result, at least should not use imagenet mean and var


