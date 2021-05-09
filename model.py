import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

import torchvision.models as models




pretrained_model = models.resnet50(pretrained=True)

class Net(nn.Module):

    input_dim = pretrained_model.inplanes

    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()

        self.dim = (im_height, im_width)

        self.pretrained_model = pretrained_model

        final_layer_input_dim = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(final_layer_input_dim, num_classes)

        for name, param in self.pretrained_model.named_parameters():
          if name != "fc":
            param.requires_grad_(False)


    def forward(self, x):
        return self.pretrained_model(x)
