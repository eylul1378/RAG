"""
Java Eğitmeni - Streamlit arayüzü.

Tamamen çevrimdışı çalışan, Think Java kitabı ve Algoritma ve Programlama ders
slaytlarıyla beslenen uzman bir Java eğitmeni. Embedding ve sohbet modelleri
Microsoft Foundry Local üzerinden bu makinede çalıştırılır, hiçbir soru veya
belge içeriği internete gönderilmez.

Not: Bilgi tabanı önceden (arka planda) hazırlanır -- bu arayüzde belge
yükleme/işleme kontrolü YOKTUR. Yeni belge eklemek için ingest.py script'i
ayrıca çalıştırılır.
"""
import streamlit as st

from config import DEFAULT_EXPLANATION_LEVEL, EXPLANATION_LEVELS, build_system_prompt
from database import count_chunks, get_top_chunks, init_db
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

# --- Kenar çubuğu: yalnızca kişiselleştirme ayarları (belge işleme kontrolü YOK) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    explanation_level = st.radio(
        "Anlatım Seviyesi",
        options=list(EXPLANATION_LEVELS.keys()),
        index=list(EXPLANATION_LEVELS.keys()).index(DEFAULT_EXPLANATION_LEVEL),
        key="explanation_level",
        help="Yanıtların ne kadar teknik/detaylı olacağını belirler.",
    )

    st.divider()
    chunk_count = count_chunks()
    if chunk_count > 0:
        st.caption(f"📚 Bilgi tabanı hazır — {chunk_count} belge parçası indekslendi.")
    else:
        st.caption("📚 Bilgi tabanı henüz hazırlanmadı.")

# --- Sohbet geçmişi ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Java hakkında bir soru sor...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        top_chunks = []

        if count_chunks() == 0:
            answer = (
                "Bilgi tabanı henüz hazır değil. Lütfen dersin sorumlusuyla iletişime geçip "
                "kaynak belgelerin (Think Java, ders slaytları) yüklenmesini bekle."
            )
            st.markdown(answer)
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
                    #    seviyesini Foundry Local'da çalışan yerel LLM'e gönderip
                    #    cevabı alıyoruz.
                    system_prompt = build_system_prompt(st.session_state.explanation_level)
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
