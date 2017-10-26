import numpy as np
import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
import skimage.io as io

epsilon = 0.000001

def predict(net, url):
    I = io.imread(url)
    image = mx.nd.array(I).astype(np.uint8)

    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))

    image = mx.image.color_normalize(image.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))
    image = mx.nd.transpose(image.astype('float32'), (2,1,0))
    image = mx.nd.expand_dims(image, axis=0)
    out = mx.nd.SoftmaxActivation(net(image))
    return (out[0][1][0].asscalar())

def predict_avg(url):
    ice_berg_net = models.resnet18_v1(pretrained=False, classes=2)
    ice_berg_net.load_params('dense-19.params', mx.cpu())
    ice_berg_net2 = models.resnet18_v1(pretrained=False, classes=2)
    ice_berg_net2.load_params('dense-34.params', mx.cpu())    
    ice_berg_net3 = models.resnet18_v1(pretrained=False, classes=2)
    ice_berg_net3.load_params('dense-39.params', mx.cpu())
    res = ((predict(ice_berg_net, url) + predict(ice_berg_net, url) + predict(ice_berg_net, url))/3)

    if res < epsilon:
        return epsilon;
    if res > (1 - epsilon):
        return (1 - epsilon);
    else:
        return res;


f = open('../input/test_list.lst', 'r')
out = open('pred.csv' , 'w')
out.write('id,is_iceberg\n')
count = 0
for line in f:
    if not count % 100:
        print('predict' + str(count) + '-th test cases');
    count +=1

    lines = line.split('\t')
    out.write((lines[2][5:13]) + ',' +str(predict_avg('../input/'+lines[2][:-1])) +'\n')

f.close()
out.close()
