import torch
import torch.nn as nn


class SimpleCNNReg(nn.Module):
    """
    Model 2: SimpleCNN dengan regularisasi untuk mengurangi overfitting.
    Perubahan dari Model 1:
      1. BatchNorm2d setelah setiap Conv  -> training lebih stabil, efek regularisasi ringan
      2. Dropout2d di blok konvolusi       -> menjatuhkan seluruh feature map secara acak
      3. Dropout sebelum Linear            -> mencegah fc menghafal data latih
    Bentuk tensor sama seperti Model 1 (input 100x100 -> output 6 kelas).
    Dilatih dengan train.py yang sama seperti model lain (Adam, NLLLoss, data dari getData.py);
    pengaturan latihannya ada di entri "model2" pada config.py.
    """

    def __init__(self, input_c=3, output=6, dropout_conv=0.1, dropout_fc=0.5):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(dropout_conv),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(dropout_conv),
        )

        self.dropout = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(50 * 9 * 9, output)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)            # (B, 20, 32, 32)
        x = self.convBlock2(x)            # (B, 50, 9, 9)
        x = self.dropout(torch.flatten(x, 1))
        x = self.fc(x)
        return self.logSoftmax(x)


if __name__ == "__main__":
    # Uji cepat: python Models/CNN2.py
    model = SimpleCNNReg()
    model.eval()                          # dropout mati saat uji
    dummy = torch.rand(2, 3, 100, 100)
    with torch.no_grad():
        out = model(dummy)
    print("Bentuk output :", tuple(out.shape))                      # (2, 6)
    print("Jumlah peluang:", out.exp().sum(dim=1))                  # ~1.0
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))
