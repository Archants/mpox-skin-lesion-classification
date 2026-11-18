# Daftar semua model beserta pengaturannya.
# Untuk menambah model: buat file model di Models/, impor di sini, lalu tambahkan satu entri baru.
#
# Keterangan setiap kunci:
#   model      : class model (harus menerima input (B, 3, H, W) dan mengeluarkan log-probabilitas 6 kelas)
#   model_args : argumen yang diberikan saat membuat model
#   img_size   : ukuran gambar setelah di-resize
#   batch_size : jumlah gambar per batch
#   epoch      : jumlah epoch per fold
#   lr         : learning rate untuk optimizer Adam
#   patience   : early stopping; berhenti jika valid loss tidak membaik selama N epoch (None = tidak dipakai)

from Models.CNN import SimpleCNN
from Models.CNN2 import SimpleCNNReg
from Models.ConvNeXt import ConvNeXtTiny

CONFIGS = {
    # Model 1: SimpleCNN sesuai materi
    "model1": {
        "model": SimpleCNN,
        "model_args": {"output": 6},
        "img_size": 100,
        "batch_size": 32,
        "epoch": 100,
        "lr": 0.001,
        "patience": None,
    },

    # Model 2: SimpleCNN + regularisasi (BatchNorm2d, Dropout2d, Dropout sebelum Linear)
    "model2": {
        "model": SimpleCNNReg,
        "model_args": {"output": 6},
        "img_size": 100,
        "batch_size": 16,
        "epoch": 30,
        "lr": 0.0001,
        "patience": 10,
    },

    # Model 3 (FINAL): ConvNeXt-Tiny pretrained ImageNet (transfer learning)
    # Pengaturan = percobaan A (model3_lr3e-5), terbaik di Valid fold 1 di antara percobaan A-D.
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
