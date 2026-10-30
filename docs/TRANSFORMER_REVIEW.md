# Transformer PR öncesi kontrol — 31 Ekim 2026

İnceleme tabanı: `origin/main = 987a18bd6515ba0e3370b582c39c342626423fe5`.
Branch: `feature/ragtruth-transformer`. Clone ve son fetch başarılı.
Tam eğitim, commit, push veya Pull Request açılmadı.

## Kontrol edilen ve commit'e hazırlanan dosyalar

1. `transformer.py`
2. `scripts/transformer.py`
3. `scripts/smoke_transformer.py`
4. `tests/test_transformer.py`
5. `configs/transformer.json`
6. `configs/jobs/transformer_train.json`
7. `requirements-transformer.txt`
8. `.github/workflows/transformer.yml`
9. `docs/TRANSFORMER.md`
10. `docs/TRANSFORMER_PR.md`
11. `docs/TRANSFORMER_REVIEW.md`
12. `README.md` — mevcut main içeriğine yalnız sekiz satır eklenmiştir.

AGENTS, ortak veri/hizalama/windowing sözleşmesi, CRF scorer ve `run_job`
arayüzleri de uyumluluk için incelendi. `ragtruth_data/`, CRF, mevcut testler,
split manifesti ve `scripts/run_job.py` değiştirilmedi.

## Mimari ve sözleşme kontrolü

| Kontrol | Sonuç |
|---|---|
| ModernBERT | AutoModel ile pretrained encoder, yeni dört logitli head; gerçek encoder ağırlıkları kontrol edildi |
| Dört tür/örtüşme | `LABEL_TYPES` sırası, multi-hot hedefler, bağımsız sigmoid; `[1,0,1,0]` korunuyor |
| Loss | Yalnız cevap ve attention maskelerinin kesişiminde BCE; kaynak/özel/padding dışlanıyor |
| Boş batch | Differentiable sıfır loss; AdamW adımı atlanıyor |
| Kaynak bağlamı | Attention içinde korunuyor; kaynak değişince cevap logits'i değişiyor |
| Uzun kaynak/cevap | Ortak pencere kodu; sentetik uzun/Unicode ve sekiz örnekte kapsam testleri geçti |
| Offset/JSON | Orijinal cevap `[start,end)` koordinatları; `{id,spans}`; farklı tür örtüşmeleri korunuyor |
| Tahmin/gold ayrımı | Tahmin yolunda gold alanlarına erişim engelleyen test geçti |
| Checkpoint | v2 safetensors + tokenizer + config + dosya SHA-256; CPU roundtrip birebir |
| Resmî test | Frozen `run_job` kapısı korunuyor; hiçbir resmî test tahmini yapılmadı |

`best` eğitim ve `min` çıkarım DATA_WINDOWING önerisine uygundur. Bunlar
farklı aşamalardaki sezgisel kararlardır: eğitimde metinsel benzerlikle tek
kaynak penceresi seçilir; çıkarımda tüm pencereler değerlendirilir. Gold ile
pencere seçimi yapılmaz. `best` gerçek kanıtı kaçırabilir, `min` ilgisiz bir
pencerenin düşük skoruyla yanlış negatif üretebilir. Validation sonuçları
olmadan bu politikanın başarılı olduğu söylenemez. Minimum validation BCE
ile checkpoint seçimi ve sabit 0.5 eşik değişmedi; ortak F1/eşik protokolü
ekipte netleştirilmelidir.

## Bulunan ve düzeltilen eksikler

**Manifest grup tutarlılığı:** Önceki loader yalnız kayıtların source/split
üyeliğine bakıyordu; eğitimde sonradan yapılan grup kontrolü seçilen kayıtlarla
sınırlıydı, predict yolunda ise yoktu. Aynı group_id farklı split'lerdeyse
sınırlı okumada bu durum kaçabiliyordu. Artık manifestin tamamı, quality/limit
filtresinden önce doğrulanıyor; yinelenen source_id ve geçersiz üyeler de
reddediliyor. Ortak veri işleme koduna dokunulmadı.

**Checkpoint dosya bütünlüğü:** Eski yükleyici sürüm ve etiket sırasını
doğruluyor, fakat ağırlık/tokenizer dosyalarının kayıttan sonra değişmesini
kontrol etmiyordu. Artık v2 checkpoint tüm model dosyalarının SHA-256 değerlerini
kaydediyor; değişmiş, eksik veya fazladan dosya varsa encoder yüklenmeden
hata veriyor. Geçerli safetensors dosyasının head bias'ı değiştirilerek regresyon
testi yapıldı. Bu checksum kontrolü kötü niyetli, birlikte yeniden yazılmış
manifestlere karşı imza doğrulaması iddiası taşımaz.

İki regresyon testi düzeltmeden önce başarısız, düzeltmeden sonra başarılı oldu.
İlk ZIP'in v1 checkpoint formatı yeni loader tarafından kabul edilmez; orijinal
kod korunur. Bu görevde eğitimden üretilmiş bir v1 checkpoint veya kullanıcı
dosyası silinmedi. Dokümanlardaki eski "Git clone yok / gerçek ağırlık denenmedi"
durumları da güncellendi.

## Test sonuçları

Ortam: Windows, Python 3.12.10, PyTorch 2.9.1+cpu, Transformers 4.57.1,
Tokenizers 0.22.2. ZIP klasöründeki mevcut `.venv` interpreter'ı kullanıldı;
test edilen kodlar yeni clone klasöründen import edildi.

| Koşum | Toplam | Geçti | Atlandı | Başarısız |
|---|---:|---:|---:|---:|
| Orijinal ZIP Transformer testleri | 18 | 18 | 0 | 0 |
| Orijinal ZIP tüm proje | 93 | 90 | 3 | 0 |
| Clone'a ilk aktarım, tüm proje | 93 | 90 | 3 | 0 |
| Yeni iki regresyon, düzeltmeden önce | 20 | 18 | 0 | 2 |
| Son Transformer suite | 20 | 20 | 0 | 0 |
| Son tüm proje suite | 95 | 92 | 3 | 0 |

Atlananlar `PreparedDataTests` içindeki count/manifest, orijinal offset/overlap
ve word-tag kapsam testleridir. `data/raw/response.jsonl` ve işlenmiş tam veri
bulunmadığından çalıştırılamadılar; **başarılı sayılmadılar**. Ham veri indirilip
resmî test etiketleri incelenmedi. Ortak nihai evaluator main'de henüz yoktur;
mevcut CRF karakter scorer'ıyla sentetik format uyumu test edildi.

`pip check` geçti. `git diff --check` kontrolü temiz. GitHub Actions workflow'u
yerelde incelendi; henüz GitHub üzerinde çalıştırılmadı.

## Gerçek ModernBERT CPU testi

Komut:

```powershell
python scripts/smoke_transformer.py --model-dir .cache/modernbert-pretrained
```

Sabit revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
İndirilen safetensors SHA-256:
`340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.

- Gerçek pretrained 22 katman / 149.014.272 encoder parametresi yüklendi.
- Embedding ve son attention ağırlıkları kaynak safetensors ile birebir eşleşti.
- İki sentetik cevap: girdi `[2,19]`, logits `[2,19,4]`, 13 cevap tokenı.
- FP32 loss `0.7538282871`; yalnız tanısal değerdir, başarı metriği değildir.
- Forward, backward, encoder gradientleri ve bir SGD optimizer adımı geçti.
- Loss dışındaki logit gradientleri sıfır; encoder gradientleri sonlu ve sıfırdan farklı.
- Ortak JSON üretildi; kaydedilip yüklenen checkpoint logits'leri birebir eşleşti.
- Smoke içindeki ölçüm yaklaşık 3,44 saniye; indirme/Python başlangıcı dahil değil,
  tam eğitim süresi veya GPU performansı tahmini değildir.

Classifier head yeni başlatılmıştır; bir adımdan sonra kullanılabilir bir
hallucination detector elde edildiği iddia edilmez. CUDA/ROCm, 2048-token tam
encoder koşumu, tam veri eğitimi ve nihai değerlendirme yapılmadı.

## Koruma, Git ve son adımlar

Yeni clone:
`C:\Users\Adnan\Desktop\ragtruth-hallucination-detection-main\ragtruth-hallucination-detection`.

Orijinal ZIP klasöründeki on teslim dosyasının SHA-256 değerleri yedekle
karşılaştırıldı; tamamı değişmeden duruyor. Ek yedek:
`../ragtruth-hallucination-detection-main/artifacts/pr-review-original-backup.zip`.
Yeni düzeltmeler yalnız clone'a uygulandı.

Branch tabanı, yerel main ve son fetch'teki origin/main aynıdır: `987a18b`.
README'ye ekleme yapıldı; mevcut main satırları korunur. Yalnız yukarıdaki
Transformer kaynak/doküman dosyaları commit kapsamındadır. Model ağırlıkları,
checkpoint, veri, ZIP ve test logları Git'e eklenmez. Model ve test checkpoint
yollarının `.gitignore` tarafından dışlandığı `git check-ignore` ile doğrulandı.

Sonraki adımlar: hazırlanmış diff'i inceleyip commit oluşturmak; kullanıcının
onayından sonra branch'i push edip `docs/TRANSFORMER_PR.md` açıklamasıyla PR
açmak. Bu inceleme sırasında commit, push ve PR işlemi yapılmadı.

Yerel kanıtlar (Git'e alınmaz): `artifacts/pr_review_original*` kayıtları ZIP
klasöründe; yeni clone'da `artifacts/pr_review_regressions_before_fix.log`,
`artifacts/pr_review_regressions_after_fix.log`, `artifacts/pr_review_all_tests.log`,
`artifacts/pr_review_transformer_tests.log`, `artifacts/pr_review_pretrained_cpu.log`
ve `artifacts/pretrained-cpu-smoke/report.json`.
