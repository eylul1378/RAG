"""
Microsoft Foundry Local entegrasyonu.

Bu modül, hem embedding (vektörleştirme) hem de sohbet/üretim (chat/generation)
modelleriyle konuşmak için ortak bir katman sağlar. ingest.py (belgeleri
vektörleştirirken) ve app.py (kullanıcı sorgusunu vektörleştirip cevap
üretirken) bu modülü paylaşır; böylece Foundry Local modelleri iki kez
farklı şekillerde yüklenmez.

Foundry Local, indirilen modelleri yerel bir OpenAI-uyumlu HTTP servisi
üzerinden sunar (FoundryLocalManager.endpoint -> http://127.0.0.1:<port>/v1).
Bu yüzden resmi `openai` Python istemcisini, base_url'i yerel servise
işaret edecek şekilde kullanabiliyoruz. Hiçbir istek internete çıkmaz.
"""
from functools import lru_cache

from openai import OpenAI

from foundry_local import FoundryLocalManager

from config import CHAT_MODEL_ALIAS, EMBEDDING_MODEL_ALIAS


@lru_cache(maxsize=1)
def get_embedding_manager() -> FoundryLocalManager:
    """Embedding modelini indirir (ilk seferde), yükler ve Foundry Local servisini başlatır.

    lru_cache sayesinde bu işlem uygulama ömrü boyunca yalnızca bir kez yapılır.
    """
    return FoundryLocalManager(EMBEDDING_MODEL_ALIAS)


@lru_cache(maxsize=1)
def get_chat_manager() -> FoundryLocalManager:
    """Sohbet/üretim modelini indirir (ilk seferde), yükler ve Foundry Local servisini başlatır."""
    return FoundryLocalManager(CHAT_MODEL_ALIAS)


def _client_for(manager: FoundryLocalManager) -> OpenAI:
    # Foundry Local, OpenAI Chat Completions / Embeddings API'siyle uyumlu bir
    # yerel endpoint sunduğu için standart OpenAI istemcisini yerel adrese
    # yönlendiriyoruz. api_key gerçek bir anahtar değil; yerel servis
    # tarafından görmezden gelinir, sadece istemcinin şart koştuğu bir alan.
    return OpenAI(base_url=manager.endpoint, api_key=manager.api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Bir metin listesini Foundry Local embedding modeliyle vektörlere çevirir."""
    manager = get_embedding_manager()
    model_id = manager.get_model_info(EMBEDDING_MODEL_ALIAS).id
    client = _client_for(manager)

    response = client.embeddings.create(model=model_id, input=texts)
    # API, girdiyle aynı sırada dönmeyebileceğinden index'e göre sıralıyoruz.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_query(text: str) -> list[float]:
    """Tek bir sorgu metnini vektöre çevirir (embed_texts'in tekil kısayolu)."""
    return embed_texts([text])[0]


def generate_answer(system_prompt: str, context_chunks: list[str], question: str) -> str:
    """Bulunan bağlam parçalarını ve kullanıcı sorusunu yerel LLM'e gönderir, cevabı döndürür."""
    manager = get_chat_manager()
    model_id = manager.get_model_info(CHAT_MODEL_ALIAS).id
    client = _client_for(manager)

    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        user_content = f"Bağlam:\n{context_text}\n\nSoru: {question}"
    else:
        # Hiç bağlam bulunamadıysa modeli yine de bilgilendiriyoruz ki
        # "bilmiyorum" cevabını verebilsin, uydurma bir cevap üretmesin.
        user_content = f"Bağlam bulunamadı.\n\nSoru: {question}"

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content
