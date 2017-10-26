import random

train = open('train.lst', 'r')
train_split = open('train_split.lst','w')
val = open('val.lst','w')

for line in train:
    if (random.random() > 0.8):
        val.write(line)
    else:
        train_split.write(line)
