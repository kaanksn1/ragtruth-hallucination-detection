# Ekip başlangıcı

Depo: https://github.com/kaanksn1/ragtruth-hallucination-detection

Önce GitHub bildirimindeki ekip davetini kabul edin. Depo başlangıçta private;
yalnız davet kabul eden üyeler erişebilir. Dört davet 30 Ekim 2026'da gönderildi.

```powershell
git clone https://github.com/kaanksn1/ragtruth-hallucination-detection.git
cd ragtruth-hallucination-detection
git switch -c feature/ragtruth-data
python -m unittest discover -s tests -v
```

Komuttaki branch'i kendi satırınızdaki adla değiştirin:

| Üye | Branch | İlk teslim |
|---|---|---|
| Serdar — `kaanksn1` | `feature/ragtruth-integration` | Veri indirme, koşum kayıtları ve GPU/entegrasyon |
| Samed — `SmtOrtakaya` | `feature/ragtruth-data` | Kaynak bazlı validation manifesti, veri/offset sözleşmesi |
| Kemal — `kemalarif` | `feature/ragtruth-baseline` | CRF kodu, küçük işlev kontrolü ve koşum komutu |
| Adnan — `adnan-demir` | `feature/ragtruth-transformer` | Model/pencereleme kodu, küçük işlev kontrolü ve pilot komutu |
| Ozan — `serhandenizhan` | `feature/ragtruth-evaluation` | Sentetik örneklerle scorer kontrolü, sonra grafik ve sunum |

İsim sırası P2 veri ve CRF teslimlerine göre düzeltilmiştir. Ayrıntılı iş tanımları
[TEAM.md](TEAM.md) içinde. Herkes kendi kodunu ve testini kendi branch'inde
geliştirir; dosyalarını açık adlarla stage edip commit/push yapar ve `main`e
PR açar. Başkasının yaptığı işi kendi katkınız gibi kaydetmeyin.

```powershell
git add kendi-dosyaniz.py
git commit -m "Describe your completed change"
git push -u origin feature/ragtruth-data
```

PR'da kullanılan veri kimliği, bağımlılıklar, komut/config ve bilinen sınırlar
bulunmalı. Serdar inceleyip birleştirir. Bütün tam eğitim ve model tahminleri
(CPU CRF dahil) Serdar'ın bilgisayarında sırayla çalışır. Üyeler kendi kodunu
geliştirir, küçük testini yapar, dönen logları inceler ve hatasını düzeltir.

Ham veri, büyük checkpoint, `.venv`, API anahtarı veya kişisel dosya yüklemeyin.
Hazırlanmış validation manifesti ve raporlar küçük, sürümlü dosyalar olmalı.
Veri ve CRF kodları teslim edildi. CRF'nin 300/100 kısa merkezî doğrulaması
geçti; tam eğitim ve model sahibinin nihai analizi bekliyor. Transformer kodu
teslim edildi ve incelendi; gerçek model GPU pilotu/tam eğitim bekliyor.
Ortak değerlendirme kodu 5. kişiden gelecek.
