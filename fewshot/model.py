import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class TemplateBank(nn.Module):
    def __init__(self, num_templates, output_channels, kernel_size):
        super(TemplateBank,self).__init__()
        self.num_templates = num_templates
        templates = [torch.Tensor(output_channels, 1, kernel_size, kernel_size) 
                                                for i in range(num_templates)]
        self.templates = nn.Parameter(torch.stack(templates))
        nn.init.orthogonal_(self.templates)

    def forward(self, coefficients):
        return (self.templates*coefficients).sum(0)

class SConv2d(nn.Module):
     def __init__(self, input_channel, bank, stride=1, padding=1):
         super(SConv2d, self).__init__()
         self.stride, self.padding, self.bank = stride, padding, bank
         self.coefficients = nn.Parameter(torch.nn.init.orthogonal_(torch.zeros(
                                     (bank.num_templates,1,input_channel,1,1))))

     def forward(self,input):
         parameters = self.bank(self.coefficients)
         return F.conv2d(input, parameters, stride=self.stride, 
                                          padding=self.padding)

class soft_conv_block(nn.Module):
    def __init__(self, bank, in_channels, out_channels, kernel_size=3,
                        stride=1, padding=1, 
                        bias=True):
                           
        super(soft_conv_block, self).__init__()
        self.stride, self.padding, self.bank = stride, padding, bank
        self.conv = SConv2d(in_channels, self.bank, stride=self.stride, 
                                                        padding=self.padding)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self,x,params=None):
        x = self.conv(x)
        x = self.bn(x)
        return x

def linear_block(in_channels, out_channels, **kwargs):
    return nn.Sequential(OrderedDict([
      ('Linear', nn.Linear(in_channels, out_channels, bias=True)),
      ('BatchNorm1d', nn.BatchNorm1d(out_channels, track_running_stats=False)), 
      ('ReLU', nn.ReLU())]))   

def conv_block(hnet_model, in_channels, out_channels, **kwargs):
    
    if hnet_model is not None:
        return soft_conv_block(hnet_model, in_channels, out_channels, **kwargs)
    else:
        return nn.Sequential(OrderedDict([
        ('Conv2d', nn.Conv2d(in_channels, out_channels, **kwargs)),
        ('BatchNorm2d', nn.BatchNorm2d(out_channels,track_running_stats=False))]))

def activation(stride, max_pool_r = 2):
    if stride == 1:
        return nn.Sequential(OrderedDict([
            ('ReLU', nn.ReLU()),
            ('MaxPool2d', nn.MaxPool2d(max_pool_r))]))
    else:
        return nn.Sequential(OrderedDict([
            ('ReLU', nn.ReLU())]))


class LinearModel(nn.Module):
    def __init__(self, out_features,
                       feedback_alignment=False,
                       momentum=1.):
        super(LinearModel, self).__init__()
        self.in_channels=784
        self.out_features=out_features
        self.feedback_alignment = feedback_alignment
        self.momentum = momentum

        self.layers = nn.Sequential()
                    
        self.layers.add_module('Layer_1', linear_block(self.in_channels, 256))
        self.layers.add_module('Layer_2', linear_block(256, 128))
        self.layers.add_module('Layer_3', linear_block(128, 64))
        self.layers.add_module('Layer_4', linear_block(64, 64))
        
        self.classifier = nn.Linear(64, out_features, bias=True)

    def forward(self, inputs):
        features = self.layers(inputs.view(-1, 784))
        return self.classifier(features)
        
class ConvModel(nn.Module):
    def __init__(self, in_channels, out_features, 
                       hidden_size=64, feature_size=64, 
                       stride=1,
                       Omniglot=True,
                       hnet_model=False, 
                       num_templates = 100,
                       momentum=1.):

        super(ConvModel, self).__init__()
        self.in_channels=in_channels
        self.out_features=out_features
        self.hidden_size=hidden_size
        self.feature_size=feature_size
        self.hnet_model = hnet_model
        self.num_templates = num_templates
        self.stride = stride
        self.Omniglot = Omniglot

        self.conv_layers_first = nn.Sequential()
        self.conv_layers_second = nn.Sequential()
        
        # hnet model
        if self.hnet_model:
            print("Creating HyperNetwork bank with # templates: ", 
                  self.num_templates)
            bank = TemplateBank(self.num_templates, hidden_size, kernel_size=3)
        else:
            bank = None

        self.padding = 1
        # first layer
        self.conv_layers_first.add_module('Layer_1', conv_block(bank, 
                        in_channels, hidden_size, kernel_size=3,
                        stride=self.stride, padding=self.padding, 
                        bias=True))

        self.conv_layers_first.add_module('Activation_1', 
                        activation(self.stride))
        
        # second layer
        self.conv_layers_first.add_module('Layer_2', conv_block(bank, 
                        hidden_size, hidden_size,  kernel_size=3,
                        stride=self.stride, padding=self.padding, 
                        bias=True))

        self.conv_layers_first.add_module('Activation_2', 
                        activation(self.stride))
        
        if self.stride == 2:
            #this is from: https://github.com/aravindr93/imaml_dev
            self.conv_layers_first.add_module('pad2',nn.ZeroPad2d((0, 1, 0, 1)))

        # third layer
        self.conv_layers_first.add_module('Layer_3', conv_block(bank, 
                        hidden_size, hidden_size, kernel_size=3,
                        stride=self.stride, padding=self.padding, 
                        bias=True))
                    
        self.conv_layers_second.add_module('Activation_3', 
                        activation(self.stride))
        # forth layer
        self.conv_layers_second.add_module('Layer_4', conv_block(bank,
                        hidden_size, hidden_size, kernel_size=3,
                        stride=self.stride, padding=self.padding,
                        bias=True))

        self.conv_layers_second.add_module('Activation_4', 
                                    activation(self.stride))

        if self.Omniglot == True:
            if self.stride == 1:
                self.classifier = nn.Linear(self.feature_size, 
                                                out_features, bias=True)
            else:
                self.classifier = nn.Linear(self.feature_size*2*2,
                                                out_features, bias=True)
        else:
            self.classifier = nn.Linear(self.feature_size, 
                                                out_features, bias=True)

    def forward(self, inputs):
        features = self.conv_layers_first(inputs)
        features = self.conv_layers_second(features)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits
