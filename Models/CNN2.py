import torch
import torch.nn as nn


class SimpleCNNReg(nn.Module):

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
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.dropout(torch.flatten(x, 1))
        x = self.fc(x)
        return self.logSoftmax(x)


if __name__ == "__main__":
    # Uji cepat: python Models/CNN2.py
    model = SimpleCNNReg()
    model.eval()
    dummy = torch.rand(2, 3, 100, 100)
    with torch.no_grad():
        out = model(dummy)
    print("Bentuk output :", tuple(out.shape))
    print("Jumlah peluang:", out.exp().sum(dim=1))
    print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))
