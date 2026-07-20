"""
Java Eğitmeni - Streamlit arayüzü.

Tamamen çevrimdışı çalışan, İngilizce kaynaklarla (Oracle Java dokümantasyonu,
Think Java kitabı, GitHub mülakat soru bankaları) beslenen ama HER ZAMAN
TÜRKÇE yanıt veren bir Java eğitmeni. Embedding ve sohbet modelleri Microsoft
Foundry Local üzerinden bu makinede çalıştırılır, hiçbir soru veya belge
içeriği internete gönderilmez.

Üç anlatım modu vardır:
- Bebek Adımları / Akademik Mod: normal retrieve-then-answer akışı, sadece
  üslup/derinlik değişir (build_system_prompt).
- Mülakat Senaryosu: farklı bir etkileşim modeli -- eğitmen önce bir soru
  sorar (INTERVIEW_QUESTION_PROMPT), kullanıcı cevaplayınca o cevabı aynı
  referans bağlama göre değerlendirir (INTERVIEW_EVALUATION_PROMPT).

Not: Bilgi tabanı önceden (arka planda) hazırlanır -- bu arayüzde belge
yükleme/işleme kontrolü YOKTUR. Yeni belge eklemek için ingest.py script'i
ayrıca çalıştırılır.
"""
import streamlit as st

from config import (
    DEFAULT_EXPLANATION_MODE,
    EXPLANATION_MODES,
    INTERVIEW_EVALUATION_PROMPT,
    INTERVIEW_MODE,
    INTERVIEW_QUESTION_PROMPT,
    build_system_prompt,
)
from database import count_chunks, get_random_chunk, get_top_chunks, init_db
from foundry_client import generate_answer

st.set_page_config(page_title="Java Eğitmeni", page_icon="☕")

init_db()

# --- Java temalı görünüm: turuncu + lacivert vurgular ---
st.markdown(
    """
    <style>
    :root {
        --java-orange: #F89820;
        --java-navy: #2A5B84;
    }
    h1, h2, h3 { color: var(--java-navy); }
    [data-testid="stSidebar"] {
        border-right: 3px solid var(--java-orange);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--java-orange);
    }
    .stChatInput, [data-testid="stChatInput"] {
        border: 1px solid var(--java-orange) !important;
    }
    div[data-testid="stChatMessage"] {
        border-left: 3px solid var(--java-navy);
        padding-left: 0.6rem;
    }
    hr { border-color: var(--java-orange) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("☕ Java Eğitmeni")
st.caption("OOP, kalıtım, kurucu metodlar, stack/heap ve koleksiyon yapıları üzerine uzmanlaşmış yerel yapay zeka eğitmeni")

# --- Oturum durumu (sidebar ve sohbet akışı ikisi de kullanır) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_chunk" not in st.session_state:
    st.session_state.interview_chunk = None

# --- Kenar çubuğu: yalnızca kişiselleştirme ayarları (belge işleme kontrolü YOK) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    explanation_mode = st.radio(
        "Anlatım Modu",
        options=list(EXPLANATION_MODES.keys()),
        index=list(EXPLANATION_MODES.keys()).index(DEFAULT_EXPLANATION_MODE),
        key="explanation_mode",
        help="Bebek Adımları: çok basit. Akademik: teknik derinlik. Mülakat Senaryosu: eğitmen sana soru sorar.",
    )

    if explanation_mode == INTERVIEW_MODE:
        st.divider()
        if st.button("🎯 Yeni Mülakat Sorusu", use_container_width=True):
            # Öncelikle dosya adında "interview" geçen bir kaynaktan seç
            # (GitHub mülakat soru bankası); yoksa herhangi bir chunk'tan seç.
            chunk = get_random_chunk(source_contains="interview")
            if chunk is None:
                st.warning("Mülakat sorusu üretmek için bilgi tabanında henüz belge yok.")
            else:
                with st.spinner("Soru hazırlanıyor..."):
                    try:
                        reference_context = [f"[Kaynak: {chunk['source']}]\n{chunk['content']}"]
                        posed_question = generate_answer(
                            INTERVIEW_QUESTION_PROMPT,
                            reference_context,
                            "Bu bilgiye dayanarak bana tek bir mülakat sorusu sor.",
                        )
                    except Exception as exc:
                        posed_question = f"Bir hata oluştu: {exc}"
                        chunk = None

                st.session_state.interview_chunk = chunk
                st.session_state.messages.append({"role": "assistant", "content": posed_question})
                st.rerun()

    st.divider()
    chunk_count = count_chunks()
    if chunk_count > 0:
        st.caption(f"📚 Bilgi tabanı hazır — {chunk_count} belge parçası indekslendi.")
    else:
        st.caption("📚 Bilgi tabanı henüz hazırlanmadı.")

# --- Sohbet geçmişi ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder = "Mülakat sorusuna cevabını yaz..." if explanation_mode == INTERVIEW_MODE else "Java hakkında bir soru sor..."
question = st.chat_input(placeholder)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        top_chunks = []

        if count_chunks() == 0:
            answer = (
                "Bilgi tabanı henüz hazır değil. Lütfen dersin sorumlusuyla iletişime geçip "
                "kaynak belgelerin yüklenmesini bekle."
            )
            st.markdown(answer)

        elif explanation_mode == INTERVIEW_MODE:
            if st.session_state.interview_chunk is None:
                # Henüz bir mülakat sorusu sorulmamış -- kullanıcıyı butona yönlendir.
                answer = (
                    "Mülakata başlamak için kenar çubuğundaki "
                    "'🎯 Yeni Mülakat Sorusu' butonuna tıkla."
                )
                st.markdown(answer)
            else:
                with st.spinner("Cevabın değerlendiriliyor..."):
                    try:
                        chunk = st.session_state.interview_chunk
                        reference_context = [f"[Kaynak: {chunk['source']}]\n{chunk['content']}"]
                        answer = generate_answer(
                            INTERVIEW_EVALUATION_PROMPT,
                            reference_context,
                            f"Adayın cevabı: {question}",
                        )
                    except Exception as exc:
                        answer = f"Bir hata oluştu: {exc}"

                st.markdown(answer)
                # Değerlendirme tamamlandı; bir sonraki soru için buton bekleniyor.
                st.session_state.interview_chunk = None
                st.caption("Yeni bir soru için kenar çubuğundaki '🎯 Yeni Mülakat Sorusu' butonuna tıkla.")

        else:
            with st.spinner("İlgili kaynak parçaları aranıyor ve yanıt hazırlanıyor..."):
                try:
                    # 1) Retrieval: soruyu Foundry Local embedding modeliyle
                    #    vektörleştirip SQLite'taki en alakalı parçaları buluyoruz.
                    top_chunks = get_top_chunks(question)
                    # Her parçayı kaynak etiketiyle birlikte gönderiyoruz ki model
                    # yanıtın sonuna doğru "Kaynak: ..." bilgisini ekleyebilsin.
                    context_texts = [
                        f"[Kaynak: {chunk['source']}]\n{chunk['content']}" for chunk in top_chunks
                    ]

                    # 2) Generation: bulunan bağlamı, soruyu ve seçilen anlatım
                    #    modunu Foundry Local'da çalışan yerel LLM'e gönderip
                    #    cevabı alıyoruz (model İngilizce bağlamdan Türkçe yanıt üretir).
                    system_prompt = build_system_prompt(explanation_mode)
                    answer = generate_answer(system_prompt, context_texts, question)
                except Exception as exc:
                    answer = f"Bir hata oluştu: {exc}"

            st.markdown(answer)

            if top_chunks:
                with st.expander("Kullanılan kaynak parçalar"):
                    for chunk in top_chunks:
                        st.markdown(f"**Kaynak:** {chunk['source']}")
                        st.markdown(chunk["content"])
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
