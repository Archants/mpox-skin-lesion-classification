import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2


class ConvNeXtTiny(nn.Module):
    def __init__(self, output=6, pretrained=True, freeze_backbone=False, augment=False):
        super(ConvNeXtTiny, self).__init__()

        # Backbone pretrained
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        # Lapisan akhir 6 kelas
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, output)

        # Bekukan backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier[2].parameters():
                param.requires_grad = True

        # Normalisasi ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.logSoftmax = nn.LogSoftmax(dim=1)

        # Augmentasi acak
        self.augment = augment
        self._augment_tf = None
        self._augment_size = None

    def _random_augment(self, x):
        size = tuple(x.shape[-2:])
        if self._augment_tf is None or self._augment_size != size:
            self._augment_size = size
            self._augment_tf = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        return torch.stack([self._augment_tf(img) for img in x])

    def forward(self, x):
        if self.augment and self.training:
            x = self._random_augment(x)
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.logSoftmax(x)
        return x


if __name__ == "__main__":
    # Uji cepat: python Models/ConvNeXt.py
    model = ConvNeXtTiny()
    model.eval()
    dummy = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print("Bentuk input   :", tuple(dummy.shape))
    print("Bentuk output  :", tuple(out.shape))
    print("Jumlah peluang :", [round(v, 4) for v in out.exp().sum(dim=1).tolist()])
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))

    frozen = ConvNeXtTiny(freeze_backbone=True)
    print("Parameter yang dilatih (freeze_backbone=True):", sum(p.numel() for p in frozen.parameters() if p.requires_grad))

    aug = ConvNeXtTiny(augment=True)
    with torch.no_grad():
        aug.train()
        beda_train = not torch.allclose(aug(dummy), aug(dummy))
        aug.eval()
        sama_eval = torch.allclose(aug(dummy), aug(dummy))
    print("Augmentasi aktif saat train():", beda_train)
    print("Augmentasi mati saat eval()  :", sama_eval)
