# 🤖 JavaBot — Yerel RAG Java Eğitmeni

Tamamen **çevrimdışı** çalışan, Retrieval-Augmented Generation (RAG) tabanlı bir Java eğitim asistanı. Sorular Oracle'ın resmi Java dokümantasyonu, *Think Java* kitabı ve GitHub mülakat soru bankalarından oluşan bir bilgi tabanına dayanarak yanıtlanır; hiçbir veri internete gönderilmez — tüm çıkarım (embedding + sohbet) [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) ile bu makinede çalışır.

## Özellikler

- **3 farklı anlatım modu** (kenar çubuğundan seçilir, her birinin kendi rengi ve kendi bağımsız sohbeti vardır):
  - 🏆 **Bebek Adımları** — teknik terim kullanmadan, çok basit ve örnekli anlatım
  - 🎓 **Akademik Mod** — doğru terminoloji ve teknik derinlikle kapsamlı anlatım
  - ⚔️ **Mülakat Senaryosu** — eğitmen sana bir Java mülakat sorusu sorar, cevabını verirsin, aynı referans materyale göre değerlendirir
- **Mod başına bağımsız sohbet** — bir moddan diğerine geçince ekran temizlenir; önceki modun sorusu/cevabı kaybolmaz, kenar çubuğundaki "Sohbet Geçmişi" listesinden erişilebilir ve tek tek silinebilir durumda kalır
- **Canlı terminal/kod paneli** — yanıt bir veya daha fazla Java kod bloğu içeriyorsa, en UZUN (en kapsamlı/tam) örnek otomatik olarak sağdaki terminal görünümlü panelde satır numaralarıyla gösterilir
- **Kaynak gösterme** — her cevabın sonunda hangi belgeden geldiği belirtilir
- **Uydurmama garantisi** — bilgi tabanında yeterince alakalı içerik bulunamazsa (kosinüs benzerliği 0.50 eşiğinin altındaysa) model hiç çağrılmaz, doğrudan sabit bir *"bulamadım"* yanıtı verilir
- **~2 dakika içinde eksiksiz yanıt** — CPU üzerinde üretim hızındaki dalgalanmaya karşı, akış (streaming) halinde gerçek zamanlı bir kesme mekanizması + neredeyse boş kalan cevaplar için otomatik tek seferlik yeniden deneme kullanılır
- **İngilizce cevaplar, Türkçe arayüz** — bilgi kaynakları ve modelin (`phi-3.5-mini`) ana dili İngilizce olduğundan, akıcılık ve doğruluk için cevaplar İngilizce üretilir; arayüz metinleri (butonlar, etiketler) Türkçe kalır
- 🎨 Figma tasarımına birebir uyarlanmış, koyu temalı, monospace/terminal esintili 3 panelli Streamlit arayüzü

## Mimari

```
Ingestion (arka planda, ingest.py ile calistirilir):

+-------------------+      +-------------------+      +-------------------+
|     documents/    | ---> |     ingest.py     | ---> |    data/rag.db    |
|  (PDF / MD / TXT) |      |  (chunk + embed)  |      |  (SQLite vectors) |
+-------------------+      +-------------------+      +-------------------+

Sorgu akisi (uygulama calisirken, her mesajda):

+-------------------+      +-------------------+      +-------------------+      +-------------------+
|     Kullanici     | ---> |       app.py      | ---> |    database.py    | ---> | foundry_client.py |
|   (Streamlit UI)  |      |    (RAG akisi)    |      |   (cosine top-k)  |      |   (embed + chat)  |
+-------------------+      +-------------------+      +-------------------+      +-------------------+
```

**Akış:** Belgeler `ingest.py` ile parçalara (chunk) bölünüp Foundry Local'ın embedding modeliyle vektörleştirilir ve SQLite'a kaydedilir → kullanıcı soru sorduğunda aynı embedding modeliyle sorgu vektörleştirilir → `database.py` kosinüs benzerliğiyle en alakalı parçayı bulur (0.50 eşiğinin altındakiler elenir) → bulunan bağlam + soru `foundry_client.py` üzerinden yerel sohbet modeline (`phi-3.5-mini`) gönderilir → cevap akış (streaming) halinde üretilir ve `app.py` hem sohbet balonunda hem de içindeki en uzun kod bloğunu ayıklayıp sağdaki terminal panelinde gösterir.

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Çalışma zamanı | [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) (`foundry-local-sdk==1.2.3`) |
| Sohbet modeli | `phi-3.5-mini` (CPU, streaming) |
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

`documents/` klasörüne kendi Java kaynaklarını (PDF, Markdown veya TXT) koy. Depoda (yerelde, `.gitignore`'lu) şu kaynaklar kullanılıyor:

- `Think Java.pdf` — Allen Downey'nin giriş seviyesi Java/CS kitabı
- `oracle-*.md` — Oracle'ın resmi *Java Tutorials* dokümantasyonundan OOP, kalıtım, kurucu metodlar, koleksiyonlar, multithreading ve daha fazlası üzerine ~33 sayfa
- `java-interview-questions*.md` — [learning-zone/java-interview-questions](https://github.com/learning-zone/java-interview-questions) reposundan mülakat soru bankaları (genel, koleksiyonlar, multithreading, programlar)

> **Mülakat modu için önemli:** Mülakat sorusu üretiminde kullanılacak dosyaların adında **"interview"** geçmeli (`get_random_chunk` bu şekilde filtreliyor).

### 4. Belgeleri işle (ingestion)

```bash
python ingest.py
```

Bu komut `documents/` klasöründeki tüm dosyaları okur, parçalara böler, embedding'lerini hesaplar ve `data/rag.db` içine kaydeder. Belgeleri her değiştirdiğinde/eklediğinde bu script'i tekrar çalıştırman gerekir (arayüzde bir "işle" butonu **yoktur** — bilgi tabanı bilinçli olarak sadece arka planda hazırlanır). Büyük dosyalar için embedding istekleri 16'lık gruplar halinde gönderilir; bu yüzden çok belgeli bir ingestion birkaç dakika sürebilir.

### 5. Uygulamayı başlat

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

## Test Etme

Uygulama açıldıktan sonra aşağıdaki adımları takip ederek üç modu da test edebilirsin:

1. **Bebek Adımları / Akademik Mod** — kenar çubuğundan modu seç, sonra sohbet kutusuna örnek bir soru yaz:
   - `How do I create an ArrayList in Java? Give me the code.`
   - `What is the difference between an abstract class and an interface?`
   - `Explain constructors in Java with an example.`

   Beklenen: ~1-2 dakika içinde İngilizce, kaynak belirtilmiş, kod örnekli bir cevap; kod varsa sağdaki terminal panelinde en kapsamlı örnek otomatik görünür.

2. **Bağlam dışı soru (uydurmama testi)** — bilgi tabanıyla alakasız bir şey sor, örn:
   - `How do I write a for loop in Python?`

   Beklenen: LLM hiç çağrılmadan anında `"I could not find this information in the documents."` cevabı.

3. **Mülakat Senaryosu** — modu seç, kenar çubuğundaki **"🎯 Yeni Mülakat Sorusu"** butonuna tıkla, eğitmenin sorduğu soruya kendi cümlelerinle cevap yaz.

   Beklenen: Eğitmen cevabın doğru olmayan/eksik kısımlarını referans materyale göre düzeltir; soru bankasındaki hazır cevabı asla doğrudan göstermez.

4. **Mod izolasyonu ve sohbet geçmişi** — bir modda soru sorduktan sonra başka bir moda geç; önceki modun ekranının temizlendiğini ama sorunun kenar çubuğundaki "Sohbet Geçmişi"nde (🗑 ile silinebilir halde) durduğunu doğrula.

5. **Bilgi tabanı durumu** — kenar çubuğunun altında `📚 Bilgi tabanı hazır — N belge parçası indekslendi.` yazısının göründüğünü doğrula (N, `ingest.py` sonrası oluşan chunk sayısıdır).

Ayrıca (isteğe bağlı) komut satırından hızlı bir doğrulama:

```bash
python -c "from database import count_chunks; print(count_chunks())"
```

## Proje Yapısı

```
RAG/
├── app.py               # Streamlit arayüzü (3 panel), mod/sohbet/mülakat akışı
├── ingest.py             # Belge okuma, parçalama, embedding, SQLite'a kayıt
├── database.py            # SQLite CRUD + kosinüs benzerliği ile retrieval
├── foundry_client.py       # Foundry Local entegrasyonu (embedding + sohbet, streaming, zaman sınırı)
├── config.py               # Sabitler, sistem promptları, mod tanımları
├── requirements.txt
├── documents/              # Kaynak belgeler (PDF/MD/TXT) — .gitignore'da, repoya dahil değil
└── data/
    └── rag.db              # SQLite veritabanı (ingest.py tarafından oluşturulur)
```

## Nasıl Çalışır — Teknik Detaylar

- **Chunking:** Belgeler paragraf sınırlarına göre ~800 karakterlik pasajlara bölünür (küçük bir karakter örtüşmesiyle bağlam kopmasın diye); tek bir paragraf bu boyutu aşarsa (nadir ama olabiliyor) cümle sınırlarında ayrıca bölünür, aksi halde tek bir aşırı büyük parça üretim süresini ciddi şekilde uzatabiliyordu.
- **Retrieval:** Sorgu embed edilir, SQLite'taki tüm vektörlerle kosinüs benzerliği hesaplanır (küçük ölçekli koleksiyonlar için brute-force yeterli). **0.50'nin altındaki** benzerlik skorları elenir — bu eşik, açıkça alakasız sorular (~0.23–0.30) ile gerçekten alakalı sorular (~0.55+) arasında kalibre edildi.
- **Zaman sınırlı üretim:** CPU üzerinde üretim hızı bağlam boyutuna göre büyük ölçüde dalgalanabildiğinden, cevaplar **akış (streaming)** halinde üretilir ve ~100 saniyelik bir üretim süresi dolduğunda son tam cümlede kesilir (retrieval + olası ek gecikmelerle toplam süre ~2 dakikanın altında hedeflenir). Neredeyse boş kalan (şanssız bir prefill/decode dengesizliğine denk gelen) cevaplar kısa bir bekleme sonrası otomatik olarak bir kez daha denenir.
- **Çıktı temizliği:** Model bazen aynı karakteri onlarca kez tekrarlayan anlamsız bir kuyruk üretebiliyor; bu tür çıktılar tespit edilip kırpılıyor. Süre sınırı yüzünden yarıda kesilen cevaplar da mümkün olduğunca son tam cümlede kesiliyor.
- **Uydurmama:** Hem sistem promptunda hem de retrieval aşamasında (benzerlik eşiği) çift katmanlı bir koruma var. Küçük modellerin "sadece bağlamı kullan" talimatını bazen görmezden gelip kendi genel bilgisinden cevap uydurabildiği gözlemlendiği için, bağlam bulunamadığında LLM hiç çağrılmaz — yanıt doğrudan Python tarafında üretilir.
- **Terminal paneli:** Cevaptaki tüm ```` ```java ```` kod blokları ayıklanır ve en uzun olanı gösterilir; bu, adım adım küçük kod parçacıklarıyla (örn. tek satırlık bir import) birlikte sonda tam bir örnek veren cevaplarda, panelde kırpık değil kapsamlı örneğin görünmesini sağlar.

## Bilinen Sınırlamalar

- **Küçük model, kusursuz değil:** `phi-3.5-mini` küçük ve CPU'da çalışan bir modeldir; İngilizce yanıtlarda genel akıcılık/doğruluk iyi olsa da, nadiren kod bloğu içinde tek bir yabancı karakter gibi kozmetik ufak hatalar görülebiliyor.
- **CPU'da çalışıyor:** Bu makinede GPU (CUDA) hızlandırma denendi ancak execution provider kaydı başarısız olduğundan devre dışı; tüm çıkarım CPU üzerinde yapılıyor, bu yüzden yanıt süreleri donanıma bağlı olarak değişebilir.
- **Mülakat modu:** Kaynak soru bankası "soru + cevap" çiftleri içerdiğinden, model bazen cevabı da sızdırmaya çalışabiliyor; bunu engellemek için hem prompt hem de çıktıyı temizleyen bir güvenlik filtresi var, ama %100 kusursuz olmayabilir.
- **Kod bloğu talimatına uyum:** Kod istenen her soruda modelin cevabı mutlaka ```` ```java ```` bloğu içinde vermesi promptla istenir ama küçük bir modelde bu %100 garanti edilemez; nadiren düz metin içinde kalan kod, terminal panelinde görünmeyebilir.

## Veri Kaynakları ve Lisanslama

- *Think Java* — Allen B. Downey, [Creative Commons Attribution-NonCommercial-ShareAlike](https://greenteapress.com/wp/think-java-2e/) lisansıyla ücretsiz dağıtılıyor.
- Oracle Java Tutorials içerikleri — [Oracle'ın resmi dokümantasyonu](https://docs.oracle.com/javase/tutorial/), eğitim amaçlı referans olarak kullanılmıştır.
- Mülakat soruları — [learning-zone/java-interview-questions](https://github.com/learning-zone/java-interview-questions) GitHub reposu.

Bu belgeler telif/boyut nedeniyle `.gitignore` ile repodan hariç tutulmuştur; yukarıdaki adımları izleyerek yeniden indirilebilir/oluşturulabilir.
