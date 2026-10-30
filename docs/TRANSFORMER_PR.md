# Önerilen PR başlığı

feat: add context-aware multilabel ModernBERT hallucination detector

## Önerilen açıklama

Ortak RAGTruth veri sözleşmesine bağlı Transformer token sınıflandırıcısı ekler.
Bir cevap tokenı aynı anda dört hata türünden birden fazlasını taşıyabilir;
sigmoid çıktıları dört bağımsız skordur. BCE yalnız geçerli cevap tokenlarında
hesaplanır, kaynak/özel/padding tokenları dışlanır. Boş cevaplı batch'ler
AdamW güncellemesi yapmaz.

- Sabit revision ile ModernBERT-base; 2048 token, eğitimde `best` kaynak
  penceresi, tahminde tüm pencerelerin karakter koordinatlarında `min` birleşimi.
- Ortak hizalama/padding/decoder kodlarının yeniden kullanımı; veri ve CRF
  uygulamalarında değişiklik yok. Train/validation hash, manifest ve grup kontrolü.
- CPU/CUDA/ROCm CLI, token ağırlıklı gradient accumulation, validation BCE ile
  checkpoint seçimi ve tamamen yerelden yüklenebilen safetensors checkpoint.
- Ortak `{id, spans}` JSONL, resmî testte mevcut frozen-run kontrolü,
  merkezî GPU job config'i, kurulum/koşum rehberi ve ayrı offline CI workflow.
- Manifestin tamamında grup/split tutarlılığı ve checkpoint v2 dosya checksum
  kontrolleri; yerel pretrained ağırlıklarla sınırlı CPU smoke scripti.

Doğrulama: Windows/Python 3.12.10/PyTorch 2.9.1+cpu/Transformers 4.57.1 ile
`python -m unittest discover -s tests -v`: 95 test, 92 geçti, 3 veri-bağımlı
test atlandı. 20 Transformer testi geçti. Forward/backward, encoder
gradientleri, optimizer, loss maskesi, overlap/uzun pencere/Unicode kapsamı,
checkpoint roundtrip, sentetik train/predict CLI ve mevcut CRF karakter
puanlayıcısıyla çıktı uyumu sınandı. `python -m pip check` temiz.

Gerçek pretrained ModernBERT-base (22 katman) ile iki sentetik cevapta CPU
forward/backward ve tek SGD adımı da geçti. Encoder ağırlıklarının indirilen
checkpoint'e eşitliği ve kaydedilip yüklenen modelin logits'lerinin birebir
eşitliği doğrulandı. Küçük rastgele encoder testleri de korunur.

Kapsam sınırı: GPU üzerinde test veya tam eğitim yapılmadı. 2048-token tam
encoder pilotu, nihai ortak scorer veya resmî test çalıştırılmadı.
Smoke sonucu model başarı metriği değildir.
Optimizer resume yok. Validation eşiği 0.5, seçim metriği masked BCE; nihai
F1/eşik protokolü ve GPU ayarları ekip pilotundan sonra sabitlenecek.

Güncel main `987a18b` ayrı klasöre clone edildi; dosyalar
`feature/ragtruth-transformer` branch'ine taşındı ve testler orada çalıştırıldı.
README'ye yalnız ek bölüm eklendi; CRF/veri kodları değişmedi.
Orijinal ZIP kodları korundu. Değişiklikler `feature/ragtruth-transformer`
branch'inden `main` hedefli Pull Request olarak teslim edilmek üzere hazırlandı.
Model ağırlıkları, checkpointler, veri setleri, ZIP ve büyük çıktı dosyaları
değişiklik kapsamına dahil edilmedi.
