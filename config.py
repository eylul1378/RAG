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

# --- Uzman Java Eğitmeni kimliği ---
# Bilgi tabanı "Think Java" kitabı ve "Algoritma ve Programlama" ders
# slaytlarından oluşur; eğitmen özellikle OOP, kalıtım, kurucu metodlar,
# yığın/öbek (stack/heap) bellek yönetimi ve ArrayList/HashMap gibi
# koleksiyon yapıları konusunda uzmanlaşmıştır.
BASE_SYSTEM_PROMPT = (
    "Sen uzman bir Java eğitmenisin. Özellikle nesne yönelimli programlama (OOP), "
    "kalıtım (inheritance), kurucu metodlar (constructors), yığın (stack) ve öbek "
    "(heap) bellek yönetimi ile ArrayList/HashMap gibi koleksiyon yapıları konusunda "
    "derin uzmanlığın var. Öğrencilere sabırlı ve teşvik edici bir üslupla yardımcı ol.\n\n"
    "SADECE sana verilen bağlam metinlerini kullanarak yanıtla. Bilgi bağlamda yoksa "
    "kesinlikle uydurma; sadece bu bilgiyi kaynaklarda bulamadığını belirt. "
    "Yanıtının en sonuna mutlaka bilgiyi aldığın kaynağı ekle "
    "(Örn: Kaynak: Think_Java.pdf veya Algoritma_Slayt_Hafta3.pdf)."
)

# Kullanıcının kenar çubuğundan seçebileceği anlatım seviyeleri ve bunların
# sistem promptuna eklenecek karşılık gelen yönergeleri.
EXPLANATION_LEVELS = {
    "Basit ve Anlaşılır": (
        "Anlatım seviyesi: Basit ve Anlaşılır. Günlük dille, az teknik terimle, "
        "somut örnekler ve benzetmeler kullanarak, yeni başlayan birine anlatır gibi açıkla."
    ),
    "Akademik ve Detaylı": (
        "Anlatım seviyesi: Akademik ve Detaylı. Doğru teknik terminoloji, formal "
        "tanımlar ve teknik derinlik kullanarak kapsamlı bir şekilde açıkla."
    ),
}

DEFAULT_EXPLANATION_LEVEL = "Basit ve Anlaşılır"


def build_system_prompt(explanation_level: str) -> str:
    """Temel eğitmen kimliğini, kullanıcının seçtiği anlatım seviyesi
    yönergesiyle birleştirerek nihai sistem promptunu oluşturur."""
    level_instruction = EXPLANATION_LEVELS.get(explanation_level, EXPLANATION_LEVELS[DEFAULT_EXPLANATION_LEVEL])
    return f"{BASE_SYSTEM_PROMPT}\n\n{level_instruction}"
