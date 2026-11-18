from Models.CNN import SimpleCNN
from Models.CNN2 import SimpleCNNReg
from Models.ConvNeXt import ConvNeXtTiny

# Pengaturan setiap model
CONFIGS = {
    # Model 1: SimpleCNN
    "model1": {
        "model": SimpleCNN,
        "model_args": {"output": 6},
        "img_size": 100,
        "batch_size": 32,
        "epoch": 100,
        "lr": 0.001,
        "patience": None,
    },

    # Model 2: SimpleCNN + regularisasi
    "model2": {
        "model": SimpleCNNReg,
        "model_args": {"output": 6},
        "img_size": 100,
        "batch_size": 16,
        "epoch": 30,
        "lr": 0.0001,
        "patience": 10,
    },

    # Model 3: ConvNeXt-Tiny pretrained
    "model3": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True},
        "img_size": 224,
        "batch_size": 16,
        "epoch": 10,
        "lr": 0.00003,
        "patience": 5,
    },
}
