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

TOP_K = 1

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
#
# Not: Model artık TÜRKÇE değil İNGİLİZCE yanıt veriyor (bkz. proje notları --
# phi-3.5-mini ağırlıklı İngilizce eğitilmiş küçük bir model; Türkçe çeviri
# katmanı akıcılığı ciddi şekilde bozuyordu, kendi ana dilinde çok daha
# tutarlı/doğru cevaplar üretiyor). Arayüz metinleri (butonlar, etiketler,
# mod adları) de tutarlılık için tamamen İngilizce'ye çevrildi.
BASE_SYSTEM_PROMPT = (
    "You are an expert Java tutor. You have deep expertise in object-oriented "
    "programming (OOP), inheritance, constructors, stack/heap memory management, "
    "and collection types like ArrayList/HashMap. Help students in a patient and "
    "encouraging tone.\n\n"
    "Answer ONLY using the context text you are given. Even if context is "
    "provided, if it does NOT actually answer the question (e.g. it's about a "
    "different programming language or an unrelated topic), do not use it -- "
    "never make things up; simply say you could not find this information in "
    "the documents. Always end your answer with the source it came from "
    "(e.g. Source: Think_Java.pdf or java-interview-questions.md)."
)

# Kullanıcının kenar çubuğundan seçebileceği anlatım modları. "Interview
# Scenario" diğer ikisinden farklıdır: build_system_prompt() ile değil,
# aşağıdaki özel INTERVIEW_* promptlarıyla ve app.py'deki ayrı bir
# etkileşim akışıyla (soru sor -> cevabı değerlendir) çalışır.
EXPLANATION_MODES = {
    "Baby Steps": (
        "Explanation mode: Baby Steps. Explain in very simple, everyday language, "
        "minimizing technical jargon, step by step, with concrete examples/"
        "analogies. Talk as if to someone who has never coded before."
    ),
    "Academic Mode": (
        "Explanation mode: Academic. Explain thoroughly using correct technical "
        "terminology, formal definitions, and technical depth."
    ),
    "Interview Scenario": (
        "Explanation mode: Interview Simulation. (This mode is handled by a "
        "separate flow in app.py; this line only exists so it appears in the "
        "option list.)"
    ),
}

DEFAULT_EXPLANATION_MODE = "Baby Steps"
INTERVIEW_MODE = "Interview Scenario"


def build_system_prompt(explanation_mode: str) -> str:
    """Temel eğitmen kimliğini, kullanıcının seçtiği anlatım modu
    yönergesiyle birleştirerek nihai sistem promptunu oluşturur.
    Not: INTERVIEW_MODE için bu fonksiyon kullanılmaz, bkz. INTERVIEW_* promptları."""
    mode_instruction = EXPLANATION_MODES.get(explanation_mode, EXPLANATION_MODES[DEFAULT_EXPLANATION_MODE])
    return f"{BASE_SYSTEM_PROMPT}\n\n{mode_instruction}"


# --- Mülakat Senaryosu modu: iki aşamalı akış (soru sor -> cevabı değerlendir) ---
# Not: Bağlam kaynağı (GitHub soru bankası) genellikle "Soru: ... Cevap: ..."
# formatında hazır çiftler içeriyor. Modelin cevabı da kopyalayıp adaya
# göstermesini KESİNLİKLE engellemek gerekiyor, yoksa mülakatın anlamı kalmaz.
INTERVIEW_QUESTION_PROMPT = (
    "You are an experienced technical Java interviewer. The context you are given "
    "is source material that contains both a question and its answer (taken from "
    "a question bank). Based on the TOPIC of this material, ask the candidate "
    "ONE single, clear interview question.\n\n"
    "STRICTLY FORBIDDEN: writing the answer, giving hints, including a code "
    "example, or explaining. ONLY write the question -- a single sentence ending "
    "in a question mark. You will wait for the candidate's own answer."
)

INTERVIEW_EVALUATION_PROMPT = (
    "You are an experienced technical Java interviewer. You just asked the "
    "candidate a question; the context you are given is the reference material "
    "that question was based on (the source of the correct/expected answer). "
    "Compare the candidate's answer against this reference: confirm what's "
    "correct, gently correct what's missing or wrong, and add supplementary "
    "information if needed. Use ONLY the information in the context, never make "
    "things up. Use a constructive and encouraging tone, just like a real "
    "interviewer would."
)
