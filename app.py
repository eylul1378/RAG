"""
JavaBot - Streamlit arayüzü.

Tamamen çevrimdışı çalışan, İngilizce kaynaklarla (Oracle Java dokümantasyonu,
Think Java kitabı, GitHub mülakat soru bankaları) beslenen ve İNGİLİZCE yanıt
veren bir Java eğitmeni (phi-3.5-mini ağırlıklı İngilizce eğitildiği için akıcılık/
doğruluk açısından bu tercih edildi -- bkz. proje notları). Arayüz metinleri de
(butonlar, etiketler) artık tamamen İngilizce. Embedding ve sohbet modelleri
Microsoft Foundry Local üzerinden bu makinede çalıştırılır, hiçbir soru veya
belge içeriği internete gönderilmez.

Arayüz iki panelden oluşur (Figma tasarımına göre, ayrı bir terminal/kod
paneli kaldırıldı -- kod zaten sohbet balonu içinde düzgün, syntax
highlighted şekilde görünüyordu; ayrı bir kopya panel yalnızca yanlış kod
bloğu seçimi/kapanmamış fence gibi ek hata kaynağı yaratıyordu):
- Sol: mod seçimi (her modun kendi vurgu rengi vardır) ve tıklanabilir/
  silinebilir sohbet geçmişi (Claude'daki gibi: her mod BİRDEN FAZLA ayrı
  sohbet/thread tutabilir -- bir mod butonuna her tıklama o modda TEMİZ,
  yeni bir sohbet başlatır, önceki sohbet kaybolmaz, Chat History'de kendi
  satırı olarak durur ve tıklanınca geri yüklenir).
- Sağ: sohbet akışı ve soru giriş kutusu.

Üç anlatım modu vardır:
- Baby Steps / Academic Mode: normal retrieve-then-answer akışı, sadece
  üslup/derinlik değişir (build_system_prompt).
- Interview Scenario: farklı bir etkileşim modeli -- eğitmen önce bir soru
  sorar (INTERVIEW_QUESTION_PROMPT), kullanıcı cevaplayınca o cevabı aynı
  referans bağlama göre değerlendirir (INTERVIEW_EVALUATION_PROMPT).

Not: Bilgi tabanı önceden (arka planda) hazırlanır -- bu arayüzde belge
yükleme/işleme kontrolü YOKTUR. Yeni belge eklemek için ingest.py script'i
ayrıca çalıştırılır.
"""
import re

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

# Her modun kendi kimlik rengi ve ikonu var (Figma tasarımına göre).
MODE_ICONS = {
    "Baby Steps": "🏆",
    "Academic Mode": "🎓",
    "Interview Scenario": "⚔️",
}
MODE_COLORS = {
    "Baby Steps": "#5A9E5A",
    "Academic Mode": "#F8981D",
    "Interview Scenario": "#E05C5C",
}

# GitHub mülakat soru bankaları çoğunlukla hazır "Soru: ... Cevap: ..." çiftleri
# içeriyor; küçük model, sadece soru sorması gerekirken bazen cevabı da
# (kod örnekleriyle birlikte) kopyalayabiliyor -- INTERVIEW_QUESTION_PROMPT'taki
# yasak talimatına rağmen. Bu, prompt'a güvenmek yerine çıktıyı ilk "cevap
# göstergesi" işaretinde (veya kod bloğunda) kesen deterministik bir güvenlik ağı.
_ANSWER_LEAK_PATTERN = re.compile(
    r"(cevap\s*[:\-]|answer\s*[:\-]|```)", re.IGNORECASE
)


def _strip_leaked_answer(text: str) -> str:
    match = _ANSWER_LEAK_PATTERN.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


# Bilgi tabanı sadece Java kaynaklarından oluşuyor, ama ortak kavramlar (örn.
# "for loop") başka bir dilde sorulsa bile embedding benzerliği eşiğin (0.50)
# üstüne çıkabiliyor -- gözlemlenen bir örnekte "How do I write a for loop in
# Python?" sorusu Think Java'daki for-loop bölümüyle 0.57 benzerlik almış ve
# model, context Java hakkında olduğu halde kendi bilgisinden Python cevabı
# üretmiş (ret talimatını göz ardı ederek). Bu yüzden retrieval/modele
# güvenmek yerine, soruda başka bir programlama dili adı geçiyorsa LLM'i hiç
# çağırmadan deterministik olarak reddediyoruz.
_OTHER_LANGUAGE_PATTERN = re.compile(
    r"\b(python|c\+\+|c#|javascript|typescript|ruby|php|golang|rust|swift|kotlin|perl|scala|matlab)(?!\w)",
    re.IGNORECASE,
)


st.set_page_config(page_title="JavaBot", page_icon="🤖", layout="wide")

init_db()

# --- Oturum durumu ---
# Claude'daki sohbet listesi mantığı: her mod, kendi içinde BİRDEN FAZLA ayrı
# sohbet (thread) tutabilir -- conversations[mod] bir mesaj listesi değil,
# mesaj listelERİ listesidir. active_thread[mod], o modda şu an hangi
# thread'in görüntülendiğini tutar. Bir mod butonuna tıklamak her zaman yeni
# (boş) bir thread açar -- önceki thread kaybolmaz, Chat History'de kendi
# satırı olarak kalır ve oradan tıklanınca geri yüklenir.
if "conversations" not in st.session_state:
    st.session_state.conversations = {mode: [[]] for mode in EXPLANATION_MODES}
if "active_thread" not in st.session_state:
    st.session_state.active_thread = {mode: 0 for mode in EXPLANATION_MODES}
if "interview_chunk" not in st.session_state:
    st.session_state.interview_chunk = None
if "explanation_mode" not in st.session_state:
    st.session_state.explanation_mode = DEFAULT_EXPLANATION_MODE


def _start_new_thread(mode_name: str) -> None:
    """Verilen mod için yeni, boş bir sohbet başlatır (Claude'daki "New Chat"
    gibi) -- önceki thread'ler dokunulmadan kalır. Zaten boş bir thread
    açıksa (kullanıcı hiç yazmadan tekrar tıkladıysa) gereksiz yere yeni
    bir boş thread daha eklemiyoruz."""
    threads = st.session_state.conversations[mode_name]
    if threads[-1]:
        threads.append([])
    st.session_state.active_thread[mode_name] = len(threads) - 1


active_mode = st.session_state.explanation_mode
active_color = MODE_COLORS[active_mode]
active_threads = st.session_state.conversations[active_mode]
active_thread_idx = st.session_state.active_thread[active_mode]
if active_thread_idx >= len(active_threads):
    active_thread_idx = len(active_threads) - 1
    st.session_state.active_thread[active_mode] = active_thread_idx
current_messages = active_threads[active_thread_idx]

# --- Koyu, terminal/kod-editörü esintili tema (Figma tasarımına göre) ---
st.markdown(
    f"""
    <style>
    :root {{
        --bg-main: #0B162C;
        --bg-panel: #060C17;
        --border-subtle: #1E293B;
        --text-dim: #64748B;
        --active-color: {active_color};
    }}
    html, body, [class*="css"] {{
        font-family: 'Fira Code', 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    }}
    .stApp {{ background-color: var(--bg-main); }}
    [data-testid="stSidebar"] {{
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border-subtle);
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #E2E8F0;
        letter-spacing: 0.05em;
    }}
    .section-label {{
        color: var(--text-dim);
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.5rem 0;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background-color: var(--bg-main);
        color: #CBD5E1;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        text-align: left;
        padding: 0.6rem 0.9rem;
        width: 100%;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: var(--active-color);
        color: var(--active-color);
    }}
    /* Aktif mod: Streamlit'in type="primary" butonu (kind="primary") burada
       o modun kimlik rengiyle boyanır -- yalnızca bir buton aynı anda primary
       olabildiğinden bu, "aktif" durumu güvenilir şekilde gösterir. */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        border-color: var(--active-color) !important;
        color: var(--active-color) !important;
        background-color: color-mix(in srgb, var(--active-color) 14%, var(--bg-main)) !important;
    }}
    .javabot-title {{
        color: #F1F5F9;
        font-size: 1.3rem;
        font-weight: 700;
    }}
    .javabot-label {{
        color: var(--active-color);
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        margin-bottom: 0.4rem;
    }}
    /* Terminal panelinin kaldırılmasıyla sohbet alanı tüm genişliği kaplıyordu
       ve boş/geniş görünüyordu -- içeriği okunabilir bir genişlikte ortalıyoruz
       (ChatGPT/Claude'daki gibi), tıpkı bir sohbet sütunu hissi versin diye. */
    [data-testid="stMainBlockContainer"] {{
        max-width: 860px;
        margin: 0 auto;
        padding-top: 2.5rem;
    }}
    [data-testid="stChatMessage"] {{
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .empty-state {{
        text-align: center;
        margin-top: 3.5rem;
        padding: 2rem;
        border: 1px dashed var(--border-subtle);
        border-radius: 16px;
    }}
    .empty-state .emoji {{ font-size: 2.2rem; }}
    .empty-state .title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 0.6rem;
    }}
    .empty-state .subtitle {{
        color: var(--text-dim);
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }}
    [data-testid="stChatInput"], .stChatInput {{
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px;
    }}
    .history-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #CBD5E1;
        font-size: 0.85rem;
        padding: 0.3rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Kenar çubuğu: mod seçimi + silinebilir sohbet geçmişi ---
with st.sidebar:
    st.markdown('<div class="javabot-title">🤖 JavaBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)

    for mode_name, icon in MODE_ICONS.items():
        is_active = mode_name == active_mode
        if st.button(
            f"{icon}  {mode_name}",
            key=f"mode_{mode_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            # Mod butonuna her tıklama TEMİZ bir sohbet başlatır -- önceki
            # sohbet Chat History'de kalır, ekrandan sadece kalkar.
            _start_new_thread(mode_name)
            st.session_state.explanation_mode = mode_name
            st.rerun()

    if active_mode == INTERVIEW_MODE:
        st.divider()
        if st.button("🎯 New Interview Question", use_container_width=True):
            # Öncelikle dosya adında "interview" geçen bir kaynaktan seç
            # (GitHub mülakat soru bankası); yoksa herhangi bir chunk'tan seç.
            chunk = get_random_chunk(source_contains="interview")
            if chunk is None:
                st.warning("There are no documents in the knowledge base yet to generate an interview question.")
            else:
                with st.spinner("Preparing question..."):
                    try:
                        reference_context = [f"[Source: {chunk['source']}]\n{chunk['content']}"]
                        posed_question = generate_answer(
                            INTERVIEW_QUESTION_PROMPT,
                            reference_context,
                            "Based on this material, ask me a single interview question.",
                        )
                        posed_question = _strip_leaked_answer(posed_question)
                    except Exception as exc:
                        posed_question = f"An error occurred: {exc}"
                        chunk = None

                st.session_state.interview_chunk = chunk
                current_messages.append({"role": "assistant", "content": posed_question})
                st.rerun()

    st.markdown('<div class="section-label">Chat History</div>', unsafe_allow_html=True)

    # Claude'daki sohbet listesi gibi: her mod BİRDEN FAZLA sohbet/thread
    # içerebilir, her biri kendi satırı olarak listelenir (o thread'deki ilk
    # soru başlık olarak kullanılır). Bir satıra tıklamak o moda VE o
    # thread'e geçer (tıpkı Claude'da bir sohbete tıklayınca onun açılması
    # gibi); çöp kutusu sadece o thread'i siler.
    any_history = False
    for mode_name, icon in MODE_ICONS.items():
        threads = st.session_state.conversations[mode_name]
        for t_idx, thread in enumerate(threads):
            first_question = next((m["content"] for m in thread if m["role"] == "user"), None)
            if first_question is None:
                continue

            any_history = True
            is_open = mode_name == active_mode and t_idx == active_thread_idx
            label = (first_question[:24] + "…") if len(first_question) > 24 else first_question
            col_label, col_delete = st.columns([5, 1])
            with col_label:
                if st.button(
                    f"{icon} {label}",
                    key=f"hist_{mode_name}_{t_idx}",
                    use_container_width=True,
                    type="primary" if is_open else "secondary",
                ):
                    st.session_state.explanation_mode = mode_name
                    st.session_state.active_thread[mode_name] = t_idx
                    st.rerun()
            if col_delete.button("✕", key=f"del_{mode_name}_{t_idx}"):
                del threads[t_idx]
                if not threads:
                    threads.append([])
                if mode_name == active_mode:
                    st.session_state.active_thread[mode_name] = min(
                        active_thread_idx, len(threads) - 1
                    )
                    if is_open:
                        st.session_state.interview_chunk = None
                st.rerun()

    if not any_history:
        st.caption("No conversations yet.")

    st.divider()
    chunk_count = count_chunks()
    if chunk_count > 0:
        st.caption(f"📚 Knowledge base ready — {chunk_count} chunks indexed.")
    else:
        st.caption("📚 Knowledge base not ready yet.")

# --- Ana alan: tam genişlik sohbet paneli ---
st.markdown(f'<div class="javabot-label">— JAVABOT</div>', unsafe_allow_html=True)

if not current_messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="emoji">🤖</div>'
        '<div class="title">Hi, I\'m JavaBot</div>'
        '<div class="subtitle">Ask me anything about Java — OOP, inheritance, collections, '
        "memory management, and more.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

placeholder = "Type your answer to the interview question..." if active_mode == INTERVIEW_MODE else "Ask something about Java..."
question = st.chat_input(placeholder)

if question:
    current_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        top_chunks = []

        if count_chunks() == 0:
            answer = (
                "The knowledge base isn't ready yet. Please contact the course "
                "instructor and wait for the source documents to be indexed."
            )
            st.markdown(answer)

        elif active_mode == INTERVIEW_MODE:
            if st.session_state.interview_chunk is None:
                answer = (
                    "Click the '🎯 New Interview Question' button in the "
                    "sidebar to start the interview."
                )
                st.markdown(answer)
            else:
                with st.spinner("Evaluating your answer..."):
                    try:
                        chunk = st.session_state.interview_chunk
                        reference_context = [f"[Source: {chunk['source']}]\n{chunk['content']}"]
                        answer = generate_answer(
                            INTERVIEW_EVALUATION_PROMPT,
                            reference_context,
                            f"Candidate's answer: {question}",
                        )
                    except Exception as exc:
                        answer = f"An error occurred: {exc}"

                st.markdown(answer)
                st.session_state.interview_chunk = None
                st.caption("Click '🎯 New Interview Question' in the sidebar for a new question.")

        elif _OTHER_LANGUAGE_PATTERN.search(question):
            answer = "I could not find this information in the documents."
            st.markdown(answer)

        else:
            top_chunks = get_top_chunks(question)

            if not top_chunks:
                # Küçük modeller "bağlam yok, uydurma" talimatına rağmen kendi
                # genel bilgisinden cevap uydurabiliyor (test edildi, güvenilmez).
                # Bu yüzden LLM'i hiç çağırmadan -- hem daha hızlı hem %100
                # garantili -- doğrudan Python'da "bulamadım" cevabını veriyoruz.
                answer = "I could not find this information in the documents."
                st.markdown(answer)
            else:
                with st.spinner("Searching relevant sources and preparing an answer..."):
                    try:
                        context_texts = [
                            f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in top_chunks
                        ]
                        system_prompt = build_system_prompt(active_mode)
                        answer = generate_answer(system_prompt, context_texts, question)
                    except Exception as exc:
                        answer = f"An error occurred: {exc}"

                st.markdown(answer)

            if top_chunks:
                with st.expander("Sources used"):
                    for chunk in top_chunks:
                        st.markdown(f"**Source:** {chunk['source']}")
                        st.markdown(chunk["content"])
                        st.divider()

    current_messages.append({"role": "assistant", "content": answer})

    # Kenar çubuğundaki "Sohbet Geçmişi" listesi bu script çalışmasının
    # BAŞINDA (mesaj eklenmeden önce) render edildiği için, güncel halin
    # görünmesi amacıyla sayfayı yeniden çalıştırıyoruz.
    st.rerun()
