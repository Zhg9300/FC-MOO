import components as cn
import torch.nn as nn
import torchvision.models as models


class ResNet(cn.Module):
    def __init__(self, device, *args, **kwargs):
        super(ResNet, self).__init__(device)
        self.model = None
        self.input_require_shape = [3, -1, -1]
        self.target_class_num = None
        self.name = 'ResNet'
        self.pretrained = False

    def generate_model(self, input_data_shape, target_class_num, *args, **kwargs):
        self.target_class_num = target_class_num

        if self.name == 'ResNet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
            base_model = models.resnet18(weights=weights)
        elif self.name == 'ResNet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
            base_model = models.resnet50(weights=weights)
        elif self.name == 'ResNet152':
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if self.pretrained else None
            base_model = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported model name: {self.name}")

        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()

        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, target_class_num)

        self.model = base_model.to(self.device)
        self.create_Loc_reshape_list()

    def forward(self, x):
        return self.model(x)


def ResNet18(device):
    module = ResNet(device)
    module.name = 'ResNet18'
    return module


def ResNet50(device):
    module = ResNet(device)
    module.name = 'ResNet50'
    return module


def ResNet152(device):
    module = ResNet(device)
    module.name = 'ResNet152'
    return module
