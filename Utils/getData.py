import os
import cv2 as cv
import numpy as np
import torch
import kagglehub
from torch.utils.data import Dataset

# Urutan kelas (alfabetis, sama dengan urutan folder). Indeks di list ini = label kelas.
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# Pasangan gambar asli yang identik / hampir identik, tetapi tercatat dengan ID pasien berbeda
# (hasil pemeriksaan dataset: file sama persis, atau skor template matching >= 0.989).
# Karena fold dibagi berdasarkan ID pasien, kembaran ini bisa terpisah ke Train dan Valid/Test (bocor).
# Saat memuat Train, gambar yang kembarannya ada di Valid/Test pada fold yang sama dilewati,
# beserta semua versi augmented-nya. Valid dan Test tidak diubah.
# Catatan: CHP_38_01 (Chickenpox) sangat mirip dengan MKP_20_01/MKP_20_02 (Monkeypox), belum dipastikan.
TWINS = [
    ('CHP_26_01', 'CHP_62_01'),          # file identik
    ('HEALTHY_21_01', 'HEALTHY_77_01'),
    ('HFMD_96_01', 'HFMD_130_01'),
    ('CWP_31_01', 'CWP_02_01'),
]


def get_dataset_root():
    # Unduh dataset dari Kaggle (hanya sekali, berikutnya memakai cache)
    return kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")


def resize_image(image, img_size):
    # Mengecilkan (tinggi dan lebar >= img_size): INTER_AREA, merata-ratakan area piksel
    # sehingga tidak muncul aliasing (tepi bergerigi). Membesarkan: INTER_LINEAR (bawaan OpenCV).
    h, w = image.shape[:2]
    interpolation = cv.INTER_AREA if (h >= img_size and w >= img_size) else cv.INTER_LINEAR
    return cv.resize(image, (img_size, img_size), interpolation=interpolation)


def get_twins_to_skip(root, fold):
    # Nama gambar Train (tanpa ekstensi) yang kembarannya ada di Valid/Test pada fold ini
    held_out = set()
    for split in ['Valid', 'Test']:
        for class_name in CLASSES:
            folder = os.path.join(root, 'Original Images', 'Original Images', 'FOLDS', f'fold{fold}', split, class_name)
            held_out.update(os.path.splitext(f)[0] for f in os.listdir(folder))

    skip = set()
    for a, b in TWINS:
        if a in held_out:
            skip.add(b)
        if b in held_out:
            skip.add(a)
    return skip - held_out  # kalau keduanya di Valid/Test, tidak ada yang perlu dilewati dari Train


class getImageLabel(Dataset):
    def __init__(self, fold=1, split='Train', img_size=100, root=None):
        # fold     : nomor fold, 1 sampai 5
        # split    : 'Train', 'Valid', atau 'Test'
        # img_size : ukuran gambar setelah di-resize (100 untuk SimpleCNN, 224 untuk model pretrained)
        # root     : lokasi dataset; kalau kosong, diambil lewat kagglehub
        if root is None:
            root = get_dataset_root()

        self.dataset = []

        # initiate the one hot encoding for class
        to_one_hot = np.eye(len(CLASSES))

        # Valid dan Test hanya dari Original Images.
        # Train = Train asli + Train augmented.
        folders = [os.path.join(root, 'Original Images', 'Original Images', 'FOLDS', f'fold{fold}', split)]
        if split == 'Train':
            folders.append(os.path.join(root, 'Augmented Images', 'Augmented Images', 'FOLDS_AUG', f'fold{fold}_AUG', 'Train'))

        # Gambar kembar yang dilewati (hanya untuk Train)
        skip = get_twins_to_skip(root, fold) if split == 'Train' else set()
        self.skipped_twins = sorted(skip)

        for k, folder in enumerate(folders):
            is_augmented = (k == 1)  # folders[0] = asli, folders[1] = augmented (hanya Train)
            for i, class_name in enumerate(CLASSES):
                class_folder = os.path.join(folder, class_name)
                for image_name in sorted(os.listdir(class_folder)):
                    # folder augmented sudah memuat salinan gambar asli (_ORIGINAL), lewati supaya tidak dobel
                    if '_ORIGINAL' in image_name:
                        continue

                    # nama gambar asli: 'CHP_26_01.jpg' -> 'CHP_26_01', augmented 'CHP_26_01_3.jpg' -> 'CHP_26_01'
                    stem = os.path.splitext(image_name)[0]
                    base_name = stem.rsplit('_', 1)[0] if is_augmented else stem
                    if base_name in skip:
                        continue

                    image = cv.imread(os.path.join(class_folder, image_name))
                    if image is None:  # file rusak / bukan gambar
                        continue

                    # OpenCV membaca BGR, ubah ke RGB, lalu resize
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image = resize_image(image, img_size)

                    # disimpan sebagai uint8 (0-255) supaya hemat memori, dinormalisasi di __getitem__
                    self.dataset.append([image, to_one_hot[i]])

        if self.skipped_twins:
            print(f"  [getData] {split} fold {fold}: {len(self.skipped_twins)} gambar kembar dilewati "
                  f"(beserta versi augmented-nya): {', '.join(self.skipped_twins)}")

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        # normalisasi ke 0-1, bentuk gambar (tinggi, lebar, 3)
        return torch.tensor(feature / 255, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # Uji cepat: python Utils/getData.py
    root = get_dataset_root()
    print("Lokasi dataset:", root)

    # gambar kembar yang dilewati dari Train di setiap fold (tanpa membaca gambar)
    for fold in range(1, 6):
        skip = sorted(get_twins_to_skip(root, fold))
        print(f"Fold {fold}: kembar yang dilewati dari Train = {', '.join(skip) if skip else '-'}")

    for split in ['Train', 'Valid', 'Test']:
        data = getImageLabel(fold=1, split=split, img_size=100, root=root)
        labels = [int(np.argmax(label)) for _, label in data.dataset]
        per_class = {c: labels.count(i) for i, c in enumerate(CLASSES)}
        print(f"\nFold 1 - {split}: {len(data)} gambar")
        print("  per kelas:", per_class)

    feature, label = data[0]
    print("\nContoh satu data (Test):")
    print("  bentuk gambar :", tuple(feature.shape), "| min-max:", round(feature.min().item(), 3), "-", round(feature.max().item(), 3))
    print("  label one-hot :", label.tolist(), "->", CLASSES[int(label.argmax())])
