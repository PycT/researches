from torch import nn, load, tensor;
from torchvision import models, transforms;

import hydro_serving_grpc as hs;

class InceptionModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(InceptionModel, self).__init__()
        model = models.inception_v3(pretrained=pretrained)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        model.fc = nn.Linear(2048, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)
