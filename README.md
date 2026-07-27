# ☕ Java Eğitmeni — Yerel RAG Yapay Zeka Asistanı

Tamamen **çevrimdışı** çalışan, Retrieval-Augmented Generation (RAG) tabanlı bir Java eğitim asistanı. Sorular Oracle'ın resmi Java dokümantasyonu, *Think Java* kitabı ve GitHub mülakat soru bankalarından oluşan bir bilgi tabanına dayanarak yanıtlanır; hiçbir veri internete gönderilmez — tüm çıkarım (embedding + sohbet) [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) ile bu makinede çalışır.

## Özellikler

- **3 farklı anlatım modu** (kenar çubuğundan seçilir):
  - **Bebek Adımları** — teknik terim kullanmadan, çok basit ve örnekli anlatım
  - **Akademik Mod** — doğru terminoloji ve teknik derinlikle kapsamlı anlatım
  - **Mülakat Senaryosu** — eğitmen sana bir Java mülakat sorusu sorar, cevabını verirsin, aynı referans materyale göre değerlendirir
- **Kaynak gösterme** — her cevabın sonunda hangi belgeden geldiği belirtilir
- **Uydurmama garantisi** — bilgi tabanında yeterince alakalı içerik bulunamazsa (benzerlik eşiğinin altındaysa) model hiç çağrılmaz, doğrudan *"Bu bilgiyi belgelerde bulamadım."* yanıtı verilir
- **~60 saniye içinde yanıt** — CPU üzerinde üretim hızındaki dalgalanmaya karşı, sabit bir zaman sınırında akış (streaming) kesme mekanizmasıyla garanti edilir
- **Türkçe arayüz ve cevaplar** — bilgi kaynakları İngilizce olsa da eğitmen her zaman Türkçe yanıt verir
- ☕ Java temalı (turuncu/lacivert) Streamlit arayüzü

## Mimari

```
┌─────────────┐      ┌───────────────┐      ┌─────────────────┐
│  documents/  │ ---> │   ingest.py   │ ---> │  data/rag.db     │
│ PDF/MD/TXT   │      │ (chunk+embed) │      │  (SQLite)        │
└─────────────┘      └───────────────┘      └─────────────────┘
                                                      │
                                                      ▼
┌─────────────┐      ┌───────────────┐      ┌─────────────────┐
│   Kullanıcı  │ <--> │    app.py     │ <--> │  database.py     │
│ (Streamlit)  │      │ (RAG akışı)   │      │ (retrieval)       │
└─────────────┘      └───────┬───────┘      └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ foundry_client.py│
                     │  (Foundry Local) │
                     │  - embedding      │
                     │  - phi-3.5-mini   │
                     └─────────────────┘
```

**Akış:** Belgeler `ingest.py` ile parçalara (chunk) bölünüp Foundry Local'ın embedding modeliyle vektörleştirilir ve SQLite'a kaydedilir → kullanıcı soru sorduğunda aynı embedding modeliyle sorgu vektörleştirilir → kosinüs benzerliğiyle en alakalı parça(lar) bulunur → bulunan bağlam + soru, yerel sohbet modeline (phi-3.5-mini) gönderilir → cevap Türkçe olarak üretilir.

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Çalışma zamanı | [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) (`foundry-local-sdk`) |
| Sohbet modeli | `phi-3.5-mini` (CPU) |
| Embedding modeli | `qwen3-embedding-0.6b` |
| Vektör deposu | SQLite (`sqlite3`, standart kütüphane) |
| Arayüz | [Streamlit](https://streamlit.io/) |
| PDF okuma | `PyPDF2` |
| Benzerlik hesabı | `numpy` (kosinüs benzerliği, brute-force) |

## Kurulum

### 1. Foundry Local'ı kur

Windows üzerinde:

```bash
winget install Microsoft.FoundryLocal
```

Diğer platformlar için [resmi kurulum talimatlarına](https://github.com/microsoft/Foundry-Local) bakın.

### 2. Python bağımlılıklarını kur

```bash
pip install -r requirements.txt
```

### 3. Belgeleri ekle

`documents/` klasörüne kendi Java kaynaklarını (PDF, Markdown veya TXT) koy. Depoda hazır olarak şunlar bulunuyor:

- `Think Java.pdf` — Allen Downey'nin giriş seviyesi Java/CS kitabı
- `oracle-*.md` — Oracle'ın resmi *Java Tutorials* dokümantasyonundan OOP, kalıtım, kurucu metodlar ve koleksiyonlar üzerine 19 sayfa
- `java-interview-questions*.md` — [learning-zone/java-interview-questions](https://github.com/learning-zone/java-interview-questions) reposundan mülakat soru bankası

> **Mülakat modu için önemli:** Mülakat sorusu üretiminde kullanılacak dosyaların adında **"interview"** geçmeli (`get_random_chunk` bu şekilde filtreliyor).

### 4. Belgeleri işle (ingestion)

```bash
python ingest.py
```

Bu komut `documents/` klasöründeki tüm dosyaları okur, parçalara böler, embedding'lerini hesaplar ve `data/rag.db` içine kaydeder. Belgeleri her değiştirdiğinde/eklediğinde bu script'i tekrar çalıştırman gerekir (arayüzde bir "işle" butonu **yoktur** — bilgi tabanı bilinçli olarak sadece arka planda hazırlanır).

### 5. Uygulamayı başlat

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

## Proje Yapısı

```
RAG/
├── app.py               # Streamlit arayüzü ve sohbet/mülakat akışı
├── ingest.py             # Belge okuma, parçalama, embedding, SQLite'a kayıt
├── database.py            # SQLite CRUD + kosinüs benzerliği ile retrieval
├── foundry_client.py       # Foundry Local entegrasyonu (embedding + sohbet, streaming)
├── config.py               # Sabitler, sistem promptları, mod tanımları
├── requirements.txt
├── documents/              # Kaynak belgeler (PDF/MD/TXT) — .gitignore'da, repoya dahil değil
└── data/
    └── rag.db              # SQLite veritabanı (ingest.py tarafından oluşturulur)
```

## Nasıl Çalışır — Teknik Detaylar

- **Chunking:** Belgeler paragraf sınırlarına göre, ~800 karakterlik pasajlara bölünür (küçük bir karakter örtüşmesiyle bağlam kopmasın diye).
- **Retrieval:** Sorgu embed edilir, SQLite'taki tüm vektörlerle kosinüs benzerliği hesaplanır (küçük ölçekli koleksiyonlar için brute-force yeterli). **0.50'nin altındaki** benzerlik skorları elenir — bu eşik, açıkça alakasız sorular (~0.23–0.30) ile gerçekten alakalı sorular (~0.55+) arasında kalibre edildi.
- **Zaman sınırlı üretim:** CPU üzerinde üretim hızı context boyutuna göre büyük ölçüde dalgalanabildiğinden (sabit bir token sınırı bazen 50 sn'de bitip bazen 130 sn'yi aşabiliyordu), cevaplar **akış (streaming)** halinde üretilir ve ~35 saniyelik bir üretim süresi dolduğunda son tam cümlede kesilir. Bu, donanım hızından bağımsız olarak toplam yanıt süresini (retrieval + üretim) ~60 saniyenin altında tutar.
- **Uydurmama:** Hem sistem promptunda hem de retrieval aşamasında (benzerlik eşiği) çift katmanlı bir koruma var. Küçük modellerin "sadece bağlamı kullan" talimatını bazen görmezden gelip kendi genel bilgisinden cevap uydurabildiği gözlemlendiği için, bağlam bulunamadığında LLM hiç çağrılmaz — yanıt doğrudan Python tarafında üretilir.

## Bilinen Sınırlamalar

- **Türkçe akıcılık:** `phi-3.5-mini` ağırlıklı olarak İngilizce eğitilmiş küçük bir modeldir. Bilgi tabanı İngilizce kaynaklardan Türkçe yanıt üretecek şekilde tasarlandığı için doğruluk iyi olsa da, cümle kuruluşu bazen pürüzlü olabilir.
- **Kısa/yarım cevaplar:** 60 saniyelik hedefi tutturmak için üretim süresi sınırlandığından, çok kapsamlı sorularda cevap kısa kalabilir veya bir cümle ortasında (nadiren) kesilebilir.
- **CPU'da çalışıyor:** Bu makinede GPU (CUDA) hızlandırma denendi ancak execution provider kaydı başarısız olduğundan devre dışı; tüm çıkarım CPU üzerinde yapılıyor.
- **Mülakat modu:** Kaynak soru bankası "soru + cevap" çiftleri içerdiğinden, model bazen cevabı da sızdırmaya çalışabiliyor; bunu engellemek için hem prompt hem de çıktıyı temizleyen bir güvenlik filtresi var, ama %100 kusursuz olmayabilir.

## Veri Kaynakları ve Lisanslama

- *Think Java* — Allen B. Downey, [Creative Commons Attribution-NonCommercial-ShareAlike](https://greenteapress.com/wp/think-java-2e/) lisansıyla ücretsiz dağıtılıyor.
- Oracle Java Tutorials içerikleri — [Oracle'ın resmi dokümantasyonu](https://docs.oracle.com/javase/tutorial/), eğitim amaçlı referans olarak kullanılmıştır.
- Mülakat soruları — [learning-zone/java-interview-questions](https://github.com/learning-zone/java-interview-questions) GitHub reposu.

Bu belgeler telif/boyut nedeniyle `.gitignore` ile repodan hariç tutulmuştur; yukarıdaki adımları izleyerek yeniden indirilebilir/oluşturulabilir.
