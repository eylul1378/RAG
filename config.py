"""Proje genelinde paylaşılan sabitler ve ayarlar."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "data", "rag.db")

# Foundry Local model alias'ları (catalog'da bu isimlerle aranır)
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"

# Chunking ayarları: pasaj düzeyinde, ~1-3 paragraf
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

TOP_K = 3

SYSTEM_PROMPT = (
    "Aşağıdaki belgelerden alınan bağlamı kullanarak kullanıcının sorusunu yanıtla. "
    "Eğer bilgi verilen bağlamda yoksa kesinlikle uydurma ve sadece "
    '"Bu bilgiyi belgelerde bulamadım" de.'
)
