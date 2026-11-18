import os
import cv2 as cv
import numpy as np
import torch
import kagglehub
from torch.utils.data import Dataset

# Daftar kelas
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# Pasangan gambar kembar
TWINS = [
    ('CHP_26_01', 'CHP_62_01'),
    ('HEALTHY_21_01', 'HEALTHY_77_01'),
    ('HFMD_96_01', 'HFMD_130_01'),
    ('CWP_31_01', 'CWP_02_01'),
]


def get_dataset_root():
    return kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")


def resize_image(image, img_size):
    h, w = image.shape[:2]
    interpolation = cv.INTER_AREA if (h >= img_size and w >= img_size) else cv.INTER_LINEAR
    return cv.resize(image, (img_size, img_size), interpolation=interpolation)


def get_twins_to_skip(root, fold):
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
    return skip - held_out


class getImageLabel(Dataset):
    def __init__(self, fold=1, split='Train', img_size=100, root=None):
        if root is None:
            root = get_dataset_root()

        self.dataset = []

        to_one_hot = np.eye(len(CLASSES))

        # Folder sumber
        folders = [os.path.join(root, 'Original Images', 'Original Images', 'FOLDS', f'fold{fold}', split)]
        if split == 'Train':
            folders.append(os.path.join(root, 'Augmented Images', 'Augmented Images', 'FOLDS_AUG', f'fold{fold}_AUG', 'Train'))

        # Gambar kembar yang dilewati
        skip = get_twins_to_skip(root, fold) if split == 'Train' else set()
        self.skipped_twins = sorted(skip)

        for k, folder in enumerate(folders):
            is_augmented = (k == 1)
            for i, class_name in enumerate(CLASSES):
                class_folder = os.path.join(folder, class_name)
                for image_name in sorted(os.listdir(class_folder)):
                    if '_ORIGINAL' in image_name:
                        continue

                    stem = os.path.splitext(image_name)[0]
                    base_name = stem.rsplit('_', 1)[0] if is_augmented else stem
                    if base_name in skip:
                        continue

                    # Baca dan resize gambar
                    image = cv.imread(os.path.join(class_folder, image_name))
                    if image is None:
                        continue

                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image = resize_image(image, img_size)

                    self.dataset.append([image, to_one_hot[i]])

        if self.skipped_twins:
            print(f"  [getData] {split} fold {fold}: {len(self.skipped_twins)} gambar kembar dilewati "
                  f"(beserta versi augmented-nya): {', '.join(self.skipped_twins)}")

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature / 255, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # Uji cepat: python Utils/getData.py
    root = get_dataset_root()
    print("Lokasi dataset:", root)

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
