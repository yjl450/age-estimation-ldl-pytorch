from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
_C.MODEL.ARCH = "se_resnext50_32x4d"  # check python train.py -h for available models
_C.MODEL.IMG_SIZE = 224

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.EPOCHS = 80
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 128


# available models: ['alexnet', 'bninception', 'cafferesnet101', 'densenet121',
# 'densenet161', 'densenet169', 'densenet201', 'dpn107', 'dpn131', 'dpn68',
# 'dpn68b', 'dpn92', 'dpn98', 'fbresnet152', 'inceptionresnetv2', 'inceptionv3',
# 'inceptionv4', 'nasnetalarge', 'nasnetamobile', 'pnasnet5large', 'polynet',
# 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
# 'resnext101_32x4d', 'resnext101_64x4d', 'se_resnet101', 'se_resnet152',
# 'se_resnet50', 'se_resnext101_32x4d', 'se_resnext50_32x4d', 'senet154',
# 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
# 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'xception']
