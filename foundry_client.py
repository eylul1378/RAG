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
from functools import lru_cache

from foundry_local_sdk import FoundryLocalManager
from foundry_local_sdk.configuration import Configuration
from foundry_local_sdk.imodel import IModel

from config import CHAT_MODEL_ALIAS, EMBEDDING_MODEL_ALIAS

# Foundry Local, uygulama verilerini (model önbelleği, loglar) bu isimle
# bir klasörde tutar. Yalnızca harf/rakam/._- karakterlerine izin verilir.
APP_NAME = "rag-yerel-asistan"


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
        user_content = f"Bağlam:\n{context_text}\n\nSoru: {question}"
    else:
        # Hiç bağlam bulunamadıysa modeli yine de bilgilendiriyoruz ki
        # "bilmiyorum" cevabını verebilsin, uydurma bir cevap üretmesin.
        user_content = f"Bağlam bulunamadı.\n\nSoru: {question}"

    completion = model.get_chat_client().complete_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    return completion.choices[0].message.content
