import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision.models as classicalModel


__all__ = [
    'VGG_nmp', 'vgg11_nmp', 'vgg11_bn_nmp', 'vgg13_nmp', 'vgg13_bn_nmp', 'vgg16_nmp', 'vgg16_bn_nmp',
    'vgg19_bn_nmp', 'vgg19_nmp',
]


class VGG_nmp(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG_nmp, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 128, 256, 256, 512, 512, 512, 512],
    'B': [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
    'D': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'E': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}


def vgg11_nmp(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['A']), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg11(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 2:
                k_num = k_num - 1
            if k_ini > 5:
                k_num = k_num - 1
            if k_ini > 10:
                k_num = k_num - 1
            if k_ini > 15:
                k_num = k_num - 1
            if k_ini > 20:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg11_bn_nmp(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg11_bn(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 3:
                k_num = k_num - 1
            if k_ini > 7:
                k_num = k_num - 1
            if k_ini > 14:
                k_num = k_num - 1
            if k_ini > 21:
                k_num = k_num - 1
            if k_ini > 28:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg13_nmp(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['B']), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg13(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 4:
                k_num = k_num - 1
            if k_ini > 9:
                k_num = k_num - 1
            if k_ini > 14:
                k_num = k_num - 1
            if k_ini > 19:
                k_num = k_num - 1
            if k_ini > 24:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg13_bn_nmp(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg13_bn(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 6:
                k_num = k_num - 1
            if k_ini > 13:
                k_num = k_num - 1
            if k_ini > 20:
                k_num = k_num - 1
            if k_ini > 27:
                k_num = k_num - 1
            if k_ini > 34:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg16_nmp(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['D']), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg16(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 4:
                k_num = k_num - 1
            if k_ini > 9:
                k_num = k_num - 1
            if k_ini > 16:
                k_num = k_num - 1
            if k_ini > 23:
                k_num = k_num - 1
            if k_ini > 30:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg16_bn_nmp(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg16_bn(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 6:
                k_num = k_num - 1
            if k_ini > 13:
                k_num = k_num - 1
            if k_ini > 23:
                k_num = k_num - 1
            if k_ini > 33:
                k_num = k_num - 1
            if k_ini > 43:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg19_nmp(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['E']), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg19(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 4:
                k_num = k_num - 1
            if k_ini > 9:
                k_num = k_num - 1
            if k_ini > 18:
                k_num = k_num - 1
            if k_ini > 27:
                k_num = k_num - 1
            if k_ini > 36:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg19_bn_nmp(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_nmp(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        vgg = classicalModel.vgg19_bn(pretrained=True)
        pretrained_dict = vgg.state_dict()
        pretrained_dict_re = {}
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            k_num = int(k.split(".")[1])
            k_ini = k_num
            if k_ini > 6:
                k_num = k_num - 1
            if k_ini > 13:
                k_num = k_num - 1
            if k_ini > 26:
                k_num = k_num - 1
            if k_ini > 39:
                k_num = k_num - 1
            k = k.split(".")[0] + "." + str(k_num) + "." + k.split(".")[2]
            pretrained_dict_re[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict_re.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    vgg11_nmp_backbone = vgg11_nmp(pretrained=True)
    print(vgg11_nmp_backbone)
#     # vgg11_bn_nmp_backbone = vgg11_bn_nmp(pretrained=True)
#     # print(vgg11_bn_nmp_backbone)
#     # vgg13_nmp_backbone = vgg13_nmp(pretrained=True)
#     # print(vgg13_nmp_backbone)
#     # vgg13_bn_nmp_backbone = vgg13_bn_nmp(pretrained=True)
#     # print(vgg13_bn_nmp_backbone)
#     # vgg16_nmp_backbone = vgg16_nmp(pretrained=True)
#     # print(vgg16_nmp_backbone)
#     # vgg16_bn_nmp_backbone = vgg16_bn_nmp(pretrained=True)
#     # print(vgg16_bn_nmp_backbone)
#     # vgg19_nmp_backbone = vgg19_nmp(pretrained=True)
#     # print(vgg19_nmp_backbone)
#     # vgg19_bn_nmp_backbone = vgg19_bn_nmp(pretrained=True)
#     # print(vgg19_bn_nmp_backbone)