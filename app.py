"""
Yerel RAG AI Asistanı - Streamlit arayüzü.

Tamamen çevrimdışı çalışır: embedding ve sohbet modelleri Microsoft Foundry
Local üzerinden bu makinede çalıştırılır, hiçbir soru veya belge içeriği
internete gönderilmez.
"""
import streamlit as st

from config import SYSTEM_PROMPT
from database import count_chunks, get_top_chunks, init_db
from foundry_client import generate_answer
from ingest import run_ingestion

st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="📚")

init_db()

st.title("📚 Yerel RAG AI Asistanı")
st.caption("Microsoft Foundry Local ile tamamen çevrimdışı çalışan belge soru-cevap asistanı")

# --- Kenar çubuğu: belge işleme (ingestion) kontrolü ---
with st.sidebar:
    st.header("Belge Yönetimi")
    st.write(f"Veritabanında şu anda **{count_chunks()}** belge parçası (chunk) var.")
    st.write("Yeni belge eklediysen veya mevcut belgeleri güncellediysen aşağıdaki butona tıkla.")

    if st.button("📥 Belgeleri Veritabanına İşle", use_container_width=True):
        with st.spinner("Belgeler okunuyor, parçalara bölünüyor ve Foundry Local ile vektörleştiriliyor..."):
            try:
                # ingest.run_ingestion() -> documents/ klasörünü okur, parçalara
                # böler ve her parçayı Foundry Local embedding modeliyle
                # vektörleştirip SQLite'a kaydeder.
                result = run_ingestion()
            except Exception as exc:
                st.error(f"Belge işleme sırasında bir hata oluştu: {exc}")
            else:
                st.success(
                    f"{result['files_processed']} dosya işlendi, "
                    f"{result['chunks_created']} parça veritabanına kaydedildi."
                )
                if result["skipped"]:
                    st.warning(
                        "Atlanan dosyalar (desteklenmeyen tür veya boş içerik): "
                        + ", ".join(result["skipped"])
                    )

    st.divider()
    st.caption("Desteklenen dosya türleri: PDF, TXT. Dosyalarını `documents/` klasörüne koy.")

# --- Sohbet geçmişi ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Belgeler hakkında bir soru sor...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        top_chunks = []

        if count_chunks() == 0:
            answer = (
                "Veritabanında henüz hiç belge yok. Lütfen önce kenar çubuğundaki "
                "'Belgeleri Veritabanına İşle' butonuna tıkla."
            )
            st.markdown(answer)
        else:
            with st.spinner("İlgili belge parçaları aranıyor ve yanıt üretiliyor..."):
                try:
                    # 1) Retrieval: soruyu Foundry Local embedding modeliyle
                    #    vektörleştirip SQLite'taki en alakalı parçaları buluyoruz.
                    top_chunks = get_top_chunks(question)
                    context_texts = [chunk["content"] for chunk in top_chunks]

                    # 2) Generation: bulunan bağlamı ve soruyu Foundry Local'da
                    #    çalışan yerel LLM'e (ör. Phi-3.5 Mini) sistem promptuyla
                    #    birlikte gönderip cevabı alıyoruz.
                    answer = generate_answer(SYSTEM_PROMPT, context_texts, question)
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
