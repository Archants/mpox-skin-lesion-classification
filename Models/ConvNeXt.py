import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2


class ConvNeXtTiny(nn.Module):
    def __init__(self, output=6, pretrained=True, freeze_backbone=False, augment=False):
        # output          : jumlah kelas (6)
        # pretrained      : True = memakai bobot hasil latihan di ImageNet (transfer learning)
        # freeze_backbone : True = bagian pretrained dibekukan, hanya lapisan terakhir (768 -> 6) yang dilatih
        # augment         : True = augmentasi acak di dalam model, HANYA aktif saat model.train()
        super(ConvNeXtTiny, self).__init__()

        # Backbone ConvNeXt-Tiny dari torchvision. Bobot pretrained diunduh otomatis saat pertama kali dipakai.
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        # Lapisan terakhir ConvNeXt ada di classifier[2]: Linear(768 -> 1000 kelas ImageNet).
        # Diganti menjadi Linear(768 -> 6 kelas) untuk dataset Mpox.
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, output)

        # Feature extraction: bekukan semua bobot backbone, kecuali lapisan terakhir yang baru dibuat.
        # Bobot yang dibekukan tidak punya gradien, sehingga tidak diubah oleh optimizer.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier[2].parameters():
                param.requires_grad = True

        # Normalisasi ImageNet (mean dan std per channel RGB), dilakukan di dalam model
        # supaya getData.py tidak perlu diubah. register_buffer: ikut pindah ke GPU, tetapi tidak dilatih.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Keluaran berupa log-probabilitas, pasangan NLLLoss di train.py
        self.logSoftmax = nn.LogSoftmax(dim=1)

        # Augmentasi acak (on-the-fly). Dibuat saat forward pertama karena ukuran crop
        # mengikuti ukuran gambar input (img_size di config.py).
        self.augment = augment
        self._augment_tf = None
        self._augment_size = None

    def _random_augment(self, x):
        size = tuple(x.shape[-2:])
        if self._augment_tf is None or self._augment_size != size:
            self._augment_size = size
            self._augment_tf = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),                              # cermin kiri-kanan
                v2.RandomVerticalFlip(p=0.5),                                # cermin atas-bawah
                v2.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True),  # potong 80-100%, lalu perbesar lagi
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # ubah kecerahan, kontras, saturasi sedikit
            ])
        # diacak per gambar (bukan satu pengacakan untuk satu batch)
        return torch.stack([self._augment_tf(img) for img in x])

    def forward(self, x):
        # x: (B, 3, 224, 224), RGB, nilai 0-1 (dari getImageLabel)
        if self.augment and self.training:   # hanya saat latihan; mati saat model.eval() (validasi dan test)
            x = self._random_augment(x)
        x = (x - self.mean) / self.std   # normalisasi ImageNet
        x = self.backbone(x)             # (B, 6) skor mentah (logits)
        x = self.logSoftmax(x)           # (B, 6) log-probabilitas
        return x


if __name__ == "__main__":
    # Uji cepat: python Models/ConvNeXt.py
    model = ConvNeXtTiny()
    model.eval()
    dummy = torch.rand(2, 3, 224, 224)   # 2 gambar acak, 3 channel, 224 x 224
    with torch.no_grad():
        out = model(dummy)
    print("Bentuk input   :", tuple(dummy.shape))
    print("Bentuk output  :", tuple(out.shape))                         # harus (2, 6)
    print("Jumlah peluang :", [round(v, 4) for v in out.exp().sum(dim=1).tolist()])  # harus [1.0, 1.0]
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))

    frozen = ConvNeXtTiny(freeze_backbone=True)
    print("Parameter yang dilatih (freeze_backbone=True):", sum(p.numel() for p in frozen.parameters() if p.requires_grad))  # harus 4614

    aug = ConvNeXtTiny(augment=True)
    with torch.no_grad():
        aug.train()
        beda_train = not torch.allclose(aug(dummy), aug(dummy))   # input sama, augmentasi acak -> hasil berbeda
        aug.eval()
        sama_eval = torch.allclose(aug(dummy), aug(dummy))        # augmentasi mati -> hasil sama
    print("Augmentasi aktif saat train():", beda_train)   # harus True
    print("Augmentasi mati saat eval()  :", sama_eval)    # harus True
