# ANN-PyTorch: Iris Verisi ile Basit Yapay Sinir Ağı

Bu proje, PyTorch kullanılarak geliştirilmiş, Iris veri seti üzerinde çalışan 3 katmanlı tam bağlantılı (fully connected) yapay sinir ağı (ANN) modelidir. Basit bir mimariyle sınıflandırma problemini çözmek isteyenler için güzel bir başlangıç örneğidir.

## Proje Yapısı

- **`model.py`**: Yapay sinir ağı mimarisini tanımlar.
- **`train.py`**: Modelin eğitildiği dosya. Hiperparametre ayarları ve eğitim döngüsü burada yer alır.
- **`test.py`**: Eğitilen modelle yeni veriler üzerinden test yapılır.

## Model Özellikleri

- **Giriş Katmanı**: 4 boyutlu iris özellik vektörü
- **Gizli Katmanlar**: Fully connected katmanlar
- **Çıkış Katmanı**: 3 sınıf (Setosa, Versicolor, Virginica)
- **Aktivasyon Fonksiyonu**: `ReLU`, son katmanda `Softmax`
- **Kayıp Fonksiyonu**: `CrossEntropyLoss`
- **Optimizasyon**: `Adam`

## Eğitim

```bash
python train.py
