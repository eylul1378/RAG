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

# Embedding isteklerini bu boyutta gruplar halinde gönderiyoruz. Büyük bir
# dosyanın yüzlerce parçasını tek seferde göndermek, Foundry Local'ın yerel
# embedding servisinde isteğin iptal edilmesine ("Operation was cancelled")
# yol açabiliyor; küçük gruplar bu riski ortadan kaldırıyor.
EMBEDDING_BATCH_SIZE = 16

# --- Uzman Java Eğitmeni kimliği ---
# Bilgi tabanı İngilizce kaynaklardan oluşur (Oracle Java dokümantasyonu,
# Think Java kitabı, GitHub mülakat soru bankaları); eğitmen özellikle OOP,
# kalıtım, kurucu metodlar, yığın/öbek (stack/heap) bellek yönetimi ve
# ArrayList/HashMap gibi koleksiyon yapıları konusunda uzmanlaşmıştır.
BASE_SYSTEM_PROMPT = (
    "Sen uzman bir Java eğitmenisin. Özellikle nesne yönelimli programlama (OOP), "
    "kalıtım (inheritance), kurucu metodlar (constructors), yığın (stack) ve öbek "
    "(heap) bellek yönetimi ile ArrayList/HashMap gibi koleksiyon yapıları konusunda "
    "derin uzmanlığın var. Öğrencilere sabırlı ve teşvik edici bir üslupla yardımcı ol.\n\n"
    "Sana verilen bağlam metinleri İngilizce olabilir (Oracle Java dokümantasyonu, "
    "Think Java kitabı, mülakat soru bankaları gibi kaynaklardan gelir). Bağlamı "
    "anlayıp SEN HER ZAMAN TÜRKÇE yanıt ver; İngilizce metni olduğu gibi kopyalama, "
    "Türkçeye açıklayarak aktar. Kod örneklerini olduğu gibi (İngilizce değişken/metod "
    "isimleriyle) koru, sadece açıklama metnini Türkçeleştir.\n\n"
    "SADECE sana verilen bağlam metinlerini kullanarak yanıtla. Bilgi bağlamda yoksa "
    "kesinlikle uydurma; sadece bu bilgiyi kaynaklarda bulamadığını belirt. "
    "Yanıtının en sonuna mutlaka bilgiyi aldığın kaynağı ekle "
    "(Örn: Kaynak: Think_Java.pdf veya java-interview-questions.md)."
)

# Kullanıcının kenar çubuğundan seçebileceği anlatım modları. "Mülakat
# Senaryosu" diğer ikisinden farklıdır: build_system_prompt() ile değil,
# aşağıdaki özel INTERVIEW_* promptlarıyla ve app.py'deki ayrı bir
# etkileşim akışıyla (soru sor -> cevabı değerlendir) çalışır.
EXPLANATION_MODES = {
    "Bebek Adımları": (
        "Anlatım modu: Bebek Adımları. Çok basit, günlük dille, teknik terimi en aza "
        "indirerek, adım adım ve somut örnek/benzetmelerle anlat. Hiç Java bilmeyen "
        "birine anlatır gibi davran."
    ),
    "Akademik Mod": (
        "Anlatım modu: Akademik. Doğru teknik terminoloji, formal tanımlar ve teknik "
        "derinlik kullanarak kapsamlı bir şekilde açıkla."
    ),
    "Mülakat Senaryosu": (
        "Anlatım modu: Mülakat Senaryosu. (Bu mod app.py'de ayrı bir akışla yönetilir, "
        "bu satır sadece seçenek listesinde görünmesi içindir.)"
    ),
}

DEFAULT_EXPLANATION_MODE = "Bebek Adımları"
INTERVIEW_MODE = "Mülakat Senaryosu"


def build_system_prompt(explanation_mode: str) -> str:
    """Temel eğitmen kimliğini, kullanıcının seçtiği anlatım modu
    yönergesiyle birleştirerek nihai sistem promptunu oluşturur.
    Not: INTERVIEW_MODE için bu fonksiyon kullanılmaz, bkz. INTERVIEW_* promptları."""
    mode_instruction = EXPLANATION_MODES.get(explanation_mode, EXPLANATION_MODES[DEFAULT_EXPLANATION_MODE])
    return f"{BASE_SYSTEM_PROMPT}\n\n{mode_instruction}"


# --- Mülakat Senaryosu modu: iki aşamalı akış (soru sor -> cevabı değerlendir) ---
INTERVIEW_QUESTION_PROMPT = (
    "Sen deneyimli bir teknik Java mülakatçısısın. Sana verilen bağlam bir mülakat "
    "sorusu/konu materyalidir. Bu bağlama dayanarak adaya TÜRKÇE, SADECE TEK bir net "
    "mülakat sorusu sor. Soru dışında selamlama, açıklama veya başka bir şey ekleme; "
    "doğrudan soruyu yaz."
)

INTERVIEW_EVALUATION_PROMPT = (
    "Sen deneyimli bir teknik Java mülakatçısısın. Adaya az önce bir soru sordun, sana "
    "verilen bağlam o sorunun dayandığı referans materyaldir (doğru/beklenen cevabın "
    "kaynağı). Adayın cevabını bu referansla karşılaştır: doğru olan kısımları TÜRKÇE "
    "olarak onayla, eksik veya yanlış olan kısımları nazikçe düzelt, gerekirse "
    "tamamlayıcı bilgi ver. SADECE bağlamdaki bilgiyi kullan, uydurma. Yapıcı ve "
    "teşvik edici bir üslup kullan, tıpkı gerçek bir mülakatçı gibi."
)
