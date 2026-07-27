"""
Microsoft Foundry Local entegrasyonu.

Bu modül, hem embedding (vektörleştirme) hem de sohbet/üretim (chat/generation)
modelleriyle konuşmak için ortak bir katman sağlar. ingest.py (belgeleri
vektörleştirirken) ve app.py (kullanıcı sorgusunu vektörleştirip cevap
üretirken) bu modülü paylaşır; böylece FoundryLocalManager (bir singleton)
uygulama ömrü boyunca yalnızca bir kez başlatılır ve iki model de aynı
native servis üzerinden yönetilir.

Foundry Local Core, native bir kütüphane (DLL) üzerinden doğrudan bu Python
süreci içinde çalışır; ayrı bir HTTP sunucusu veya sistemde çalışan bir
"foundry" servis sürecine ihtiyaç duymaz. Bu yüzden tüm çağrılar
süreç-içi (in-process) ve tamamen çevrimdışıdır -- hiçbir istek
internete çıkmaz.
"""
import time
from functools import lru_cache

from foundry_local_sdk import FoundryLocalManager
from foundry_local_sdk.configuration import Configuration
from foundry_local_sdk.imodel import IModel

from config import CHAT_MODEL_ALIAS, EMBEDDING_MODEL_ALIAS

# Foundry Local, uygulama verilerini (model önbelleği, loglar) bu isimle
# bir klasörde tutar. Yalnızca harf/rakam/._- karakterlerine izin verilir.
APP_NAME = "rag-yerel-asistan"

# Hedef: 60 saniyenin altında yanıt. Retrieval ~7 sn sürüyor ve kesme
# kontrolü chunk'lar arasında yapıldığından birkaç saniyelik sarkma
# gözlemlendi (bkz. proje notları); bu yüzden geniş bir pay bırakıyoruz.
_MAX_GENERATION_SECONDS = 35


@lru_cache(maxsize=1)
def _get_manager() -> FoundryLocalManager:
    """Foundry Local Core'u başlatır. lru_cache + singleton kontrolü sayesinde
    bu işlem süreç başına yalnızca bir kez yapılır (Streamlit her kullanıcı
    etkileşiminde script'i yeniden çalıştırdığı için bu önemlidir)."""
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    return FoundryLocalManager.instance


def _get_ready_model(alias: str) -> IModel:
    """Alias'a karşılık gelen modeli katalogda bulur, gerekiyorsa indirir
    (ilk çalıştırmada modeller yerel diske indirilir) ve belleğe yükler."""
    manager = _get_manager()
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model '{alias}' Foundry Local katalogunda bulunamadı.")
    if not model.is_cached:
        model.download()
    if not model.is_loaded:
        model.load()
    return model


@lru_cache(maxsize=1)
def _get_embedding_model() -> IModel:
    return _get_ready_model(EMBEDDING_MODEL_ALIAS)


@lru_cache(maxsize=1)
def _get_chat_model() -> IModel:
    return _get_ready_model(CHAT_MODEL_ALIAS)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Bir metin listesini Foundry Local embedding modeliyle vektörlere çevirir."""
    model = _get_embedding_model()
    response = model.get_embedding_client().generate_embeddings(texts)
    # API, girdiyle aynı sırada dönmeyebileceğinden index'e göre sıralıyoruz.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_query(text: str) -> list[float]:
    """Tek bir sorgu metnini vektöre çevirir (embed_texts'in tekil kısayolu)."""
    return embed_texts([text])[0]


def generate_answer(system_prompt: str, context_chunks: list[str], question: str) -> str:
    """Bulunan bağlam parçalarını ve kullanıcı sorusunu yerel LLM'e gönderir, cevabı döndürür."""
    model = _get_chat_model()

    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        # Küçük modeller sistem promptundaki kuralları bazen es geçebiliyor;
        # kaynak gösterme talimatını her turda soruyla birlikte tekrarlamak
        # (sistem promptuna ek olarak) uyumu belirgin şekilde artırıyor.
        user_content = (
            f"Bağlam:\n{context_text}\n\nSoru: {question}\n\n"
            "Unutma: Yanıtının en sonuna 'Kaynak: <dosya adı>' şeklinde kaynağı ekle. "
            "Doğru ve derinlikli bir cevap ver; gerekirse örnek/kod ekle. Ama "
            "SÖYLEDİĞİNİ TEKRARLAMA -- aynı bilgiyi özet/sonuç bölümünde ikinci "
            "kez yazma, cevabı bir kez ve net şekilde ver."
        )
    else:
        # Hiç bağlam bulunamadıysa (retrieval hiçbir yeterince alakalı chunk
        # bulamadı) -- küçük modeller "sadece bağlamı kullan" kuralını görmezden
        # gelip kendi genel bilgisinden cevap uydurma eğiliminde olduğundan,
        # burada TEK kabul edilebilir yanıtı doğrudan komut olarak veriyoruz.
        user_content = (
            f"Soru: {question}\n\n"
            "Bu soruyla ilgili bilgi tabanında (Java ders materyalleri) hiçbir "
            "alakalı içerik bulunamadı. Kendi genel bilgini KULLANMA, cevap "
            "UYDURMA. SADECE şu cevabı ver, başka hiçbir şey ekleme: "
            "\"Bu bilgiyi belgelerde bulamadım.\""
        )

    chat_client = model.get_chat_client()
    # max_tokens bir üst güvenlik sınırı (aşırı uzun/sonsuz üretime karşı);
    # asıl süre garantisi aşağıdaki gerçek zamanlı (wall-clock) kesmeden gelir.
    chat_client.settings.max_tokens = 600

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _stream_with_time_limit(chat_client, messages)


def _stream_with_time_limit(chat_client, messages: list[dict]) -> str:
    """Cevabı akış (streaming) halinde alır ve _MAX_GENERATION_SECONDS dolduğunda
    üretimi durdurup o ana kadar toplananı döndürür.

    CPU üzerinde üretim hızı belirgin şekilde dalgalanabiliyor (bkz. proje
    notları): sabit bir max_tokens bazen 55 sn'de bitiyor, bazen 130 sn'yi
    aşıyordu. Gerçek zamanlı kesme, donanım hızından bağımsız olarak 60
    saniyelik hedefi garanti eder; sabit token sayısından farklı olarak CPU
    o an hızlıysa daha fazla, yavaşsa daha az içerik üretilmiş olur.
    """
    start = time.monotonic()
    collected: list[str] = []
    truncated = True

    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                collected.append(delta)
            if chunk.choices[0].finish_reason is not None:
                truncated = False
                break
        if time.monotonic() - start > _MAX_GENERATION_SECONDS:
            break

    text = "".join(collected).strip()
    if truncated:
        text = _trim_to_sentence_boundary(text)
    return text


def _trim_to_sentence_boundary(text: str) -> str:
    """Zaman sınırı yüzünden yarıda kesilen bir cevabı, mümkünse son tam
    cümlenin sonunda keserek daha temiz görünmesini sağlar."""
    best_cut = -1
    for sep in (".", "!", "?", "\n"):
        idx = text.rfind(sep)
        if idx > best_cut:
            best_cut = idx
    # Metnin en az yarısını koruyoruz; aksi halde aşırı kısaltma daha kötü olur.
    if best_cut > len(text) * 0.4:
        return text[: best_cut + 1].strip()
    return text
