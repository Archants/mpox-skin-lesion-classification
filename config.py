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

    # Model 2: SimpleCNN dengan parameter berbeda (diisi anggota Model 2)

    # Model 3: ConvNeXt-Tiny pretrained ImageNet (transfer learning)
    "model3": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True},
        "img_size": 224,      # ukuran input bawaan model pretrained ImageNet
        "batch_size": 16,     # lebih kecil dari Model 1 karena model besar, supaya muat di VRAM 8 GB
        "epoch": 20,
        "lr": 0.0001,         # 10x lebih kecil dari Model 1 supaya bobot pretrained tidak rusak dan tidak cepat overfitting
        "patience": 5,        # early stopping jika valid loss tidak membaik 5 epoch berturut-turut
    },

    # Percobaan A untuk Model 3: sama dengan model3, hanya learning rate diturunkan
    # (pada lr 0.0001 epoch terbaik sudah di epoch 1 dan train loss langsung hampir 0)
    "model3_lr3e-5": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True},
        "img_size": 224,
        "batch_size": 16,
        "epoch": 10,
        "lr": 0.00003,        # 3x lebih kecil dari model3
        "patience": 5,
    },

    # Percobaan D untuk Model 3: sama dengan model3_lr3e-5 + augmentasi acak di dalam model
    "model3_aug": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True, "augment": True},
        "img_size": 224,
        "batch_size": 16,
        "epoch": 10,
        "lr": 0.00003,
        "patience": 5,
    },

    # Percobaan C untuk Model 3: learning rate lebih kecil lagi
    "model3_lr1e-5": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True},
        "img_size": 224,
        "batch_size": 16,
        "epoch": 10,
        "lr": 0.00001,        # 10x lebih kecil dari model3
        "patience": 5,
    },

    # Percobaan B untuk Model 3: backbone dibekukan, hanya lapisan terakhir yang dilatih (feature extraction)
    "model3_freeze": {
        "model": ConvNeXtTiny,
        "model_args": {"output": 6, "pretrained": True, "freeze_backbone": True},
        "img_size": 224,
        "batch_size": 16,
        "epoch": 10,
        "lr": 0.001,          # lebih besar karena hanya lapisan terakhir (baru, dari nol) yang dilatih
        "patience": 5,
    },
}
