import torch
import torch.nn as nn
from torchvision import models


class BreastCancerClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, freeze_backbone=True):
        super(BreastCancerClassifier, self).__init__()

        self.model_name = model_name.lower()

        if self.model_name == 'resnet50':
            # Load pretrained ResNet50
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except:
                self.backbone = models.resnet50(pretrained=True)

            if freeze_backbone:
                # Freeze all layers first
                for param in self.backbone.parameters():
                    param.requires_grad = False

                # Unfreeze layer3 and layer4 for fine-tuning
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True

            # Replace the final fully connected layer
            # ResNet50's fc layer input features: 2048
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif self.model_name == 'vgg16':
            try:
                self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except:
                self.backbone = models.vgg16(pretrained=True)

            if freeze_backbone:
                for param in self.backbone.features.parameters():
                    param.requires_grad = False
                # Unfreeze last conv block
                for param in self.backbone.features[24:].parameters():
                    param.requires_grad = True

            num_ftrs = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        else:
            raise ValueError(f"Model {model_name} not supported yet.")

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # Test block
    model = BreastCancerClassifier('resnet50')
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
