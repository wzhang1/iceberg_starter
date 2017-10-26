from __future__ import print_function
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io

import mxnet as mx
from mxnet.test_utils import download
mx.random.seed(127)

training_dataset="../input/train_split.rec"
validation_dataset="../input/val.rec"

log_interval = 1
gpus = 0

# Options are imperative or hybrid. Use hybrid for better performance.
mode = 'hybrid'

# training hyperparameters
batch_size = 128

epochs = 50
learning_rate = 0.3
wd = 0




Height = 128
Width = 128


# load dataset
train_iter = mx.io.ImageRecordIter(path_imgrec=training_dataset,
                                   min_img_size=Height,
                                   data_shape=(3, Height, Width),
                                   rand_crop=True,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   max_random_scale=1.2,
                                   min_random_scale=0.8,
                                   rand_resize         = True,
                                   brightness          = 0.4,
                                   contrast            = 0.4,
                                   saturation          = 0.4,
                                   pca_noise           = 0.1,
                                   rand_mirror=True)
val_iter = mx.io.ImageRecordIter(path_imgrec=validation_dataset,
                                 min_img_size=Height,
                                 data_shape=(3, Height, Width),
                                 batch_size=batch_size)


from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


ice_berg_net = models.resnet34_v1(pretrained=False, classes=2)
ice_berg_net.initialize(mx.init.MSRAPrelu())


# return metrics string representation
def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])
metric = mx.metric.create(['acc', 'loss'])

import mxnet.gluon as gluon
from mxnet.image import color_normalize

def evaluate(net, data_iter, ctx):
    data_iter.reset()
    for batch in data_iter:
        data = color_normalize(batch.data[0]/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    out = metric.get()
    metric.reset()
    print(out)
    return out


import mxnet.autograd as autograd

def train(net, train_iter, val_iter, epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': wd})
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    best_ac = 0
    val_names, val_accs = evaluate(net, val_iter, ctx)
    logging.info('[Initial] validation: %s'%(metric_str(val_names, val_accs)))
    for epoch in range(epochs):
        tic = time.time()
        train_iter.reset()
        btic = time.time()
        for i, batch in enumerate(train_iter):
            # the model zoo models expect normalized images
            data = color_normalize(batch.data[0]/255,
                                   mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                                   std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # rescale the loss based on class to counter the imbalance problem
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if log_interval and not (i+1)%log_interval:
                names, accs = metric.get()
                logging.info('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                               epoch, i, batch_size/(time.time()-btic), metric_str(names, accs)))
            btic = time.time()

        names, accs = metric.get()
        metric.reset()
        logging.info('[Epoch %d] training: %s'%(epoch, metric_str(names, accs)))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        val_names, val_accs = evaluate(net, val_iter, ctx)
        logging.info('[Epoch %d] validation: %s'%(epoch, metric_str(val_names, val_accs)))

        if val_accs[0] > best_ac:
            best_ac = val_accs[0]
            logging.info('Best validation found. Checkpointing...')
            net.save_params('net1-%d.params'%(epoch))

if mode == 'hybrid':
    ice_berg_net.hybridize()
if epochs > 0:
    contexts = [mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()]
    ice_berg_net.collect_params().reset_ctx(contexts)
    train(ice_berg_net, train_iter, val_iter, epochs, contexts)


