# Klasifikasi Lesi Kulit Mpox dengan Deep Learning

Tugas mata kuliah Infrastruktur dan Platform untuk Sains Data: klasifikasi gambar kulit ke dalam 6 kelas
(Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox) memakai dataset
[Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20),
dievaluasi dengan 5-fold cross validation.

> Laporan hasil (metrik, grafik, dan confusion matrix ketiga model) akan ditambahkan setelah semua model selesai dilatih.

## Struktur program

```
├── Models/
│   ├── __init__.py
│   └── CNN.py            # SimpleCNN (Model 1 dan Model 2)
├── Utils/
│   ├── __init__.py
│   └── getData.py        # unduh dataset (kagglehub) dan class getImageLabel
├── Checkpoints/          # model hasil training (.pt), tidak diunggah ke GitHub
├── Results/              # grafik loss, confusion matrix, dan metrik per model
├── config.py             # daftar model beserta pengaturannya
├── train.py              # training 5 fold
├── test.py               # testing dan perhitungan metrik 5 fold
└── requirements.txt
```

## Persiapan environment

Dikembangkan dengan **Python 3.10** dan PyTorch 2.15.0.

```bash
conda create -n ipsd python=3.10 -y
conda activate ipsd
```

Install PyTorch sesuai perangkat (pilih salah satu):

```bash
# Laptop dengan GPU NVIDIA
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132

# Tanpa GPU NVIDIA (CPU saja, jauh lebih lambat)
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

Install library lainnya:

```bash
pip install -r requirements.txt
```

Cek apakah GPU terbaca (hasil `True` berarti GPU dipakai):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Tanpa GPU NVIDIA, training bisa dijalankan di Google Colab atau Kaggle Notebook (clone repositori ini di sana).

## Dataset

Dataset **tidak** disimpan di repositori. Saat `train.py` atau `test.py` dijalankan pertama kali, dataset diunduh otomatis
lewat `kagglehub` ke folder cache (`~/.cache/kagglehub/`). Unduhan hanya terjadi sekali.

- Train: `Original Images` Train + `Augmented Images` Train dari fold yang sama (file `_ORIGINAL` di folder augmented dilewati supaya tidak dobel)
- Valid dan Test: hanya dari `Original Images`

Cek cepat pembacaan data:

```bash
python Utils/getData.py
```

## Menjalankan

Semua perintah dijalankan dari folder utama repositori.

```bash
# uji coba cepat: 1 fold, 2 epoch
python train.py --model model1 --folds 1 --epochs 2
python test.py --model model1 --folds 1

# training dan testing penuh: 5 fold, epoch sesuai config.py
python train.py --model model1
python test.py --model model1
```

Hasil tersimpan di `Results/<model>/`:

| File | Isi |
|---|---|
| `training_foldK.png` | Grafik train loss dan valid loss |
| `history_foldK.csv` | Loss dan accuracy per epoch |
| `training_summary.csv` | Epoch terbaik dan valid loss terbaik setiap fold |
| `confusion_matrix_foldK.png` | Confusion matrix setiap fold |
| `confusion_matrix_all.png` | Confusion matrix gabungan semua fold |
| `classification_report_foldK.txt` | Precision, recall, dan F1 per kelas |
| `test_metrics.csv` | Accuracy, precision, recall, F1, ROC-AUC setiap fold, ditambah mean dan std |

## Menambah model

1. Buat file model di `Models/`. Model wajib menerima input `(batch, 3, tinggi, lebar)` dan mengeluarkan
   log-probabilitas 6 kelas (diakhiri `LogSoftmax`), karena training memakai `NLLLoss`.
2. Tambahkan satu entri di `CONFIGS` pada `config.py` (class model, ukuran gambar, batch size, epoch, learning rate).
3. Jalankan `python train.py --model <nama>` dan `python test.py --model <nama>`.

`train.py`, `test.py`, dan `getData.py` tidak perlu diubah.

## Alur kerja Git untuk anggota

```bash
git pull                      # selalu ambil versi terbaru sebelum mulai
git checkout -b model2        # kerjakan di branch sendiri
# ... ubah config.py, jalankan training dan testing ...
git add config.py Results/model2
git commit -m "Tambah Model 2"
git push -u origin model2     # lalu buat Pull Request di GitHub
```

Jangan mengunggah file model (`.pt`, `.pth`) maupun dataset.
