# V2 sonuçları neden yükselmedi?

Bu inceleme mevcut eğitim kayıtlarını ve önceden üretilmiş **dev** aday
tablolarını açıklar. Yeni eğitim/eşik araması yapılmadı; frozen kararlar ve
holdout sonuçları değiştirilmedi. Kodda kesin bir hata olmadığı iddia edilmiyor;
aşağıdaki ölçülebilen davranışlar ile henüz kanıtlanmamış nedenler ayrılıyor.

## 1. Öncelik verilen metrik ile kullanıcıya gösterilen ana skor farklı

Kullanıcıya en çok any-character (hatalı bölgeyi bulma) F1 anlatıldı. V2 seçim
kuralı ise önce dört türün macro-character F1'ını, yalnız eşitlikte any F1'ı
önceliklendiriyor. Tür başına bağımsız eşik seçimi any F1'ı optimize etmiyor.
Bu tercih kodda ilan edildiği gibi uygulanmış olsa da, kullanıcıya anlatılan
iyileştirme hedefini yeterince iyi dengelemiyor.

**Aynı focal epoch-2 ağırlıkları, aynı dev kümesi:**

| Decoder | Any F1 (%) | Dört-tür macro F1 (%) |
|---|---:|---:|
| Varsayılan 0.5 eşikleri | 51,22 | 25,55 |
| Dev tür-macro F1 ile seçilen eşikler | 44,96 | 29,93 |

Yalnız decoder değişimi macro'yu +4,38 yüzde puan artırırken any F1'ı -6,26
puan düşürüyor. Bu, training işleminin çalışmamasından kaynaklanan bir durum
değil; iki hedefin bu adaylarda çatışmasının doğrudan gözlemi.

Ayrıca aynı aday listesindeki focal epoch-3/0.5 any F1'ı %53,33. Bu değer,
sonradan yeni nihai model seçmek veya holdout için yeni başarı iddia etmek
amacıyla kullanılmıyor; seçim kuralının hangi adayı neden dışladığını açıklıyor.

## 2. CRF decoder'ında nadir tür için aşırı pozitif işaretleme

Seçilen structured epoch-3'te dev için:

| Decoder | Any F1 (%) | Tür macro F1 (%) |
|---|---:|---:|
| O-bias 0 | 20,33 | 12,3445 |
| Tür başına seçilen bias | 10,53 | 12,3670 |

Seçilen Subtle Conflict O-bias'ı -2. Bu ayar o tür için 830 gold karaktere
karşılık **1.843.432 karakter** işaretliyor ve recall 1.0 oluyor. O sınıfın F1'ı
yalnız yaklaşık %0,09. Macro'daki çok küçük kazanım, any F1'daki büyük kayba
karşı bir engelle karşılaşmıyor. Bu, eşitlik bozulduğu için any F1 tie-break'iyle
engellenemez. Yanlış alarm kontrolü/ana hedef kuralında açık bir zayıflık var.

Structured modelin bias-0 any F1'ı da %20,33 olduğu için kötü sonucun tamamı
eşik/bias seçimine bağlanamaz. Bu model ailesi mevcut eğitim reçetesiyle de
kontrolün gerisinde kaldı. Düşük loss tek başına iyi F1 anlamına gelmedi.

## 3. Model ve veri tarafındaki sınırlamalar sürüyor

V2 train span sayıları: Evident Conflict 3.643, Subtle Conflict **140**, Evident
Baseless Info 4.194, Subtle Baseless Info 1.786. Nadir sınıfın az desteği gerçek;
ancak bütün performans kaybının tek nedeni olduğu kanıtlanmadı.

Train loss düşerken BCE dev loss epoch-2'de 0.04660'tan epoch-3'te 0.04986'ya
yükseliyor. Focal dev loss da 0.00722 → 0.00742 → 0.00795 biçiminde kötüleşiyor.
Bu yüzden daha fazla epoch'un tek başına çözüm olduğunu destekleyen kanıt yok.
Farklı loss ailelerinin loss büyüklükleri birbiriyle doğrudan kıyaslanmaz.

Loss/CRF değişimleri ince anlam ilişkilerini öğrenme başarısını garanti etmedi.
Kaynakta hangi bilginin desteklendiği, yanlış atıf ve örtük çelişki gibi hata
türleri için temsil/öğrenme eksikliği ayrıca araştırılmalı. Bu son cümle kesin
kök-neden kanıtı değil, mevcut hata örneklerinden çıkan bir araştırma hipotezi.

## Sonraki bir geliştirme turu için çıkarım

Önce hedef metrik açıkça belirlenmeli ve aşırı yanlış alarma karşı seçim kuralı
dev üzerinde sınanmalı. Eşik değişimi ile model kapasitesi kazanımı ayrı
raporlanmalı. Bu tanı, %80'e ulaşılacağını veya yeniden eğitim gerektiğini
kanıtlamaz. Mevcut holdout'lar yeni ayar seçmek için kullanılmamalı; mevcut
sonuçlar aynen arşivde tutulmalı.

Kaynaklar: [focal adayları](focal_calibration.json),
[structured adayları](structured_calibration.json), [dev karşılaştırması](dev_comparison.md),
[train/dev sınıf desteği ve hata örnekleri](error_analysis.json).
