# Anonymous — NLP Proje 2: RAGTruth

**Amaç:** Kaynak içerik ve yapay zekâ cevabını birlikte inceleyerek cevaptaki
kaynakla çelişen veya kaynakta desteklenmeyen metin bölümlerini işaretlemek.

Örnek (açıklama için uydurulmuştur): Kaynak “Şirket 2020'de Ankara'da kuruldu”
derken cevap “Şirket 2018'de İstanbul'da kuruldu” diyorsa, sorunlu tarih ve yer
ifadelerini bulmak. Bu proje, cevabın kaynağa bağlılığını inceler; genel amaçlı
bir gerçek dünya doğrulama sistemi olduğu iddiasında bulunmaz.

Depo: [kaanksn1/ragtruth-hallucination-detection](https://github.com/kaanksn1/ragtruth-hallucination-detection).
Ekip davetleri 30 Ekim'de gönderildi; katılım için GitHub davetini kabul etmek gerekir.
Başlangıç görünürlüğü private'tır.

Bu bağımsız depo yeni projenin başlangıç alanıdır. BANKING77 dosyaları, deneyleri ve
kişisel katkı raporları bu çalışma sırasında değiştirilmedi. Henüz bir RAGTruth
modeli eğitilmedi, sunum hazırlanmadı ve resmî test değerlendirmesi yapılmadı.
29 Ekim'de sentetik girdilerle GPU forward/backward/optimizer kontrolü yapıldı;
bu işlem gerçek veriyle model eğitimi veya başarı deneyi değildir.

## Veri kaynağı ve ilk kontrol

- [Resmî GitHub deposu](https://github.com/ParticleMedia/RAGTruth)
- [ACL 2024 makalesi](https://aclanthology.org/2024.acl-long.585/)
- Sabit sürüm: `c103204b9ce28d6bbad859304bf30de72b8ed8fe`.
- Ham dosyalar ve kaynak LICENSE'ı yerel `data/raw/` klasöründe saklanır ve
  Git'e alınmaz. Sürüm bilgisi `configs/source.json` içindedir.
- SHA-256 ve tekrar üretilebilir inceleme çıktısı:
  [training_data_audit.json](docs/training_data_audit.json).

| Kontrol | Sonuç |
|---|---:|
| Resmî eğitim cevabı | 15.090 |
| Resmî test cevabı (yalnız ID/split sayımı) | 2.700 |
| Eğitim kaynak grubu | 2.515 |
| Eğitimde en az bir anotasyon içeren cevap | 6.721 |
| Eğitimde anotasyonlu bölüm | 12.756 |
| Eğitimde örtüşen bölümler içeren cevap | 101 |
| Geçersiz veya metinle uyuşmayan eğitim span konumu | 0 |
| Resmî train/test arasında ortak source_id | 0 |
| Eğitimde tam aynı cevap metinlerinin fazladan kopyaları | 211 |

Eğitimde 14.942 cevap `good`, 28 cevap `truncated`, 120 cevap
`incorrect_refusal` olarak işaretli. Tekrarlar ve kalite işaretleri raporlandı;
henüz hiçbir kayıt silinmedi. Aynı cevap metni farklı kaynaklarda geçebilir;
yalnız metne bakarak toplu silme yapılmamalı.

| Eğitim anotasyon türü | Bölüm sayısı |
|---|---:|
| Evident Baseless Info | 5.499 |
| Evident Conflict | 4.705 |
| Subtle Baseless Info | 2.367 |
| Subtle Conflict | 185 |

Etiket sayısı dört gerçek türdür. BIO veya başka bir kodlama nedeniyle artan
çıktı etiketi sayısı yeni anlamsal sınıflar gibi sunulmamalı.

Eğitim cevaplarının medyan uzunluğu 125, kaynakla birlikte kullanılan üretim
promptlarının medyan uzunluğu 357 boşlukla ayrılmış kelimedir. Promptların
yüzde 95'lik uzunluğu 917 kelimedir. **Bunlar model tokenı değildir.** Kaynak
bağlamını 512 tokenda sessizce kesmek kanıtı kaybettirebilir. Tokenizer ile
gerçek uzunluklar ve pencereleme maliyeti ayrıca ölçülmeli.

## 4–5 günlük çalışma kapsamı

Paylaşılan beş ders dosyası incelendi. CRF + kaynak bağlamlı küçük Transformer
ders içeriğiyle uyumlu; P2'ye özel zorunlu model sayısı henüz kesinleşmedi.
[Slayt eşleştirmesi ve P1/P2 ayrımı](docs/LECTURE_ALIGNMENT.md) kaydedildi.
Aşağıdaki kapsam hocanın güncel P2 yönergesine göre güncellenebilecek bir taslaktır.

1. Kaynak–cevap eşlemesini, dört türü ve karakter konumlarını koruyan ortak
   veri sözleşmesi oluşturmak.
2. Bir karşılaştırma modeli ve kaynak bağlamını kullanan küçük bir Transformer
   token sınıflandırıcısı geliştirmek. Kesin mimari, bağlam uzunluğu ve epoch
   sayısı GPU pilotundan sonra seçilecek.
3. Temel hedef olarak sorunlu bölümleri bulmak; dört türün performansını da
   incelemek. İkili bulma metriği ile türü de doğru tahmin etmeyi gerektiren
   metriği ayrı adlandırmak ve açıklamak.
4. Aynı Transformer'ın yalnız cevabı gördüğü bir kontrolü, zaman bütçesi
   yeterliyse eklemek. Böylece kaynak bağlamının etkisi daha açık incelenebilir.
5. Örnek kaynak ve cevap üzerinde tahminleri renkle gösteren yerel bir demo,
   sınıf bazında sonuçlar ve açıklanmış başarı/hata örnekleri hazırlamak.

Bu kapsamda 13B modelin tam eğitimi öngörülmüyor. Makaledeki büyük model
eğitiminin donanım ihtiyacı, bizim küçük model tasarımımızın maliyeti olarak
sunulmamalı; aynı başarı düzeyi de vaat edilmemeli.

## Görev dağılımı — 29 Ekim 2026

Kullanıcının kararı: **bütün tam eğitimler kendi bilgisayarında çalışacak;
5. kişi sunumu yapacak.** Model kodunun ve deney analizinin sorumluluğu ilgili
üyede kalır. Önceki projenin kişisel katkı raporları bu projeye taşınmaz.

| Kişi | Ana sorumluluk |
|---|---|
| 1 — Serdar / `kaanksn1` | GPU ortamı, merkezî eğitim/tahmin koşumları, entegrasyon ve küçük demo |
| 2 — Adnan / `adnan-demir` | Veri inceleme, kaynak bazlı split, ortak etiket/offset hazırlığı ve pencere eşlemesi |
| 3 — Samed / `SmtOrtakaya` | CRF baseline, kendi deney ayarları ve baseline hata analizi |
| 4 — Kemal / `kemalarif` | Kaynak bağlamlı küçük Transformer, tahmini karakter aralıklarına dönüştürme ve model analizi |
| 5 — Ozan / `serhandenizhan` | Ortak değerlendirme, 2–3 karşılaştırma grafiği, slaytları birleştirme ve sunum |

İsimler önceki projedeki sırayla eşleştirildi. Başlangıç adımları:
[ekip için GitHub rehberi](docs/GITHUB_START.md).

Herkes kendi bölümünün açıklamasını, en az üç hata örneğini (uygunsa), 1–2
slayt taslağını ve konuşmacı notunu hazırlar. 5. kişi bütün proje içeriğini
tek başına yazmaz. Her üye kendi branch/PR'ıyla katkı yapar.

Ayrıntılı teslimler, merkezî eğitim süreci ve dosya sözleşmesi:
[TEAM.md](docs/TEAM.md). Bu belge bir görev atamasıdır; kişilerin görevlerini
tamamladığını veya eğitimlerin başladığını göstermez.

## Takvim

| Gün | Ortak çıktı |
|---|---|
| 1 | 2. kişi veri/split sözleşmesini, 3–4 model iskeletlerini, 5 metrik kontrollerini hazırlar; 1 GPU pilotunu koordine eder |
| 2 | 1, hazır baseline ve Transformer kodlarını kendi PC'sinde çalıştırır; 3–4 kendi kod hatalarını giderir |
| 3 | Validation analizleri, sınırlı ayar seçimi, bütçe uygunsa kaynak bağlamı kontrolü; herkes kendi slayt taslağını verir |
| 4 | Protokolü sabitleme; 1'in PC'sinde tek resmî test değerlendirmesi; 5 sonuçları birleştirir, demo ve sunum provası yapılır |
| 5 (varsa) | Tekrar üretilebilirlik kontrolü ve süreli prova; görülen test sonuçlarına göre yeniden model seçilmez |

Pilotun amacı: gerçek modelde GPU belleğini, örnek/token hızını ve tam eğitim
süresini ölçmek. Kaynak hazırlama ve uzun bağlam stratejisi birinci gün
çözülemezse, denenen ve tamamlanan kapsam açıkça raporlanmalı.

## Donanım kontrolü — 28 Ekim 2026

Mevcut ortam salt okunur komutla kontrol edildi:

- Ubuntu-24.04 WSL, mevcut `/opt/banking77-venv` ortamı;
- Python 3.12.3, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`;
- `torch.cuda.is_available()`: `True`;
- GPU: AMD Radeon RX 7800 XT, görülen VRAM yaklaşık 15,8 GiB.

Bu kontrol model eğitimi/pilot değildir. Mevcut BANKING77 ortamına paket
eklenmedi veya sürüm değişikliği yapılmadı. P2 bağımlılıkları için ayrı ortam
veya mevcut ortamla uyumlu, kaydı tutulmuş bir kullanım yolu seçilmeli.

29 Ekim'de aynı ortamda 256 ve 512 tokenlık sentetik encoder/optimizer
kontrolleri geçti. [GPU hazırlığı ve ölçüm sınırları](docs/GPU_READINESS.md)
kaydedildi. Gerçek verili pilot, 2/4. kişilerin veri/model teslimlerini bekliyor.

## Yeniden çalıştırma

[Veri indirme ve merkezî koşum rehberi](docs/RUNNING.md) Serdar'ın entegrasyon
bölümünü açıklar. `scripts/run_job.py`, ekipten gelecek model komutlarını
sırayla çalıştırıp config/kod/veri kimliğini ve başarısızlık loglarını saklar.
Bu katman model implementasyonu değildir.

Bağımsız RAGTruth deposunun kökünden, Python 3.12 ile:

```powershell
python scripts/download_data.py
python scripts/inspect_training_data.py
python -m unittest discover -s tests -v
```

Betik yalnız standart Python kütüphanesini kullanır. Eğitim anotasyonlarını
inceler; resmî testin yalnız kayıt kimliklerini ve split üyeliğini kullanır.
Resmî test etiketlerini incelemez veya onlarla model seçmez.

## P2 tablosu için önerilen satır

- **Group Name:** Anonymous
- **Datasets:** RAGTruth — 17,790 LLM responses grounded in 2,965 source contexts;
  human-annotated hallucination spans with four types.
- **Raw Data Source URL:** https://github.com/ParticleMedia/RAGTruth
- **Downstream Task:** Context-Grounded Hallucination Span Detection and Typing
  (Token Classification)

Canlı tabloya yazılmadı; görev kontenjanı ve hocanın kabulü henüz doğrulanmadı.

## Değerlendirme disiplini

[Protokol taslağına](docs/PROTOCOL_DRAFT.md) uyulmalı. Hazır anotasyonlu veri
üzerinde çalışıldığından, veri temizliği için ekstra puan garanti edilmez.
Bu ilk inceleme sıfır hatalı span konumu buldu; gerçek olmayan temizleme emeği
iddiası eklenmemeli.
