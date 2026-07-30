"""
JavaBot - Streamlit arayüzü.

Tamamen çevrimdışı çalışan, İngilizce kaynaklarla (Oracle Java dokümantasyonu,
Think Java kitabı, GitHub mülakat soru bankaları) beslenen ve İNGİLİZCE yanıt
veren bir Java eğitmeni (phi-3.5-mini ağırlıklı İngilizce eğitildiği için akıcılık/
doğruluk açısından bu tercih edildi -- bkz. proje notları). Arayüz metinleri de
(butonlar, etiketler) artık tamamen İngilizce. Embedding ve sohbet modelleri
Microsoft Foundry Local üzerinden bu makinede çalıştırılır, hiçbir soru veya
belge içeriği internete gönderilmez.

Arayüz üç panelden oluşur (Figma tasarımına göre):
- Sol: mod seçimi (her modun kendi vurgu rengi vardır) ve silinebilir sohbet
  geçmişi.
- Orta: sohbet akışı ve soru giriş kutusu.
- Sağ: terminal görünümlü bir kod paneli -- en son yanıt kod içeriyorsa
  otomatik olarak orada gösterilir, içermiyorsa bekleme mesajı görünür.

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
# Figma tasarımında terminal panelinin arka planı da moda göre değişiyor
# (sadece vurgu rengi değil) -- Baby Steps: koyu yeşil-siyah, Academic Mode:
# standart lacivert-siyah, Interview Scenario: koyu kırmızı-siyah.
MODE_TERMINAL_BG = {
    "Baby Steps": "#05120C",
    "Academic Mode": "#060C17",
    "Interview Scenario": "#140505",
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


# Sağ paneldeki "terminal" kutusu için: yanıt bir Java kod bloğu içeriyorsa
# onu ayıklayıp orada gösteriyoruz (kullanıcının istediği davranış budur --
# terminal paneli sadece dekoratif değil, kodun gittiği yer).
_CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    """Yanıt genellikle adım adım küçük kod parçacıkları (örn. tek satırlık bir
    import) ve en sonda hepsini birleştiren TAM bir örnek içeriyor. En UZUN
    kod bloğunu seçiyoruz ki terminal panelinde küçük bir parça değil, en
    kapsamlı/kullanışlı örnek görünsün."""
    matches = _CODE_BLOCK_PATTERN.findall(text)
    blocks = [m.strip() for m in matches if m.strip()]
    if not blocks:
        return None
    return max(blocks, key=len)


def _last_code_in(messages: list[dict]) -> str | None:
    """Bir mod'un konuşmasındaki en son kod bloğunu bulur (terminal paneli için).
    Sabit bir session_state alanında tutmak yerine her render'da buradan
    türetiyoruz ki mod değişince veya bir kayıt silinince otomatik güncel kalsın."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            code = _extract_code(message["content"])
            if code:
                return code
    return None


st.set_page_config(page_title="JavaBot", page_icon="🤖", layout="wide")

init_db()

# --- Oturum durumu ---
# Her mod kendi bağımsız sohbetini tutar (conversations[mod_adı]) ki bir
# moddan diğerine geçince önceki modun konuşması ekranda kalmasın -- yeni
# seçilen mod temiz bir ekranla (veya daha önce o modda bırakılan yerden)
# başlasın, eski konuşma sadece "Sohbet Geçmişi" listesinden erişilebilsin.
if "conversations" not in st.session_state:
    st.session_state.conversations = {mode: [] for mode in EXPLANATION_MODES}
if "interview_chunk" not in st.session_state:
    st.session_state.interview_chunk = None
if "explanation_mode" not in st.session_state:
    st.session_state.explanation_mode = DEFAULT_EXPLANATION_MODE

active_mode = st.session_state.explanation_mode
active_color = MODE_COLORS[active_mode]
active_terminal_bg = MODE_TERMINAL_BG[active_mode]
current_messages = st.session_state.conversations[active_mode]

# --- Koyu, terminal/kod-editörü esintili tema (Figma tasarımına göre) ---
st.markdown(
    f"""
    <style>
    :root {{
        --bg-main: #0B162C;
        --bg-panel: #060C17;
        --bg-terminal: {active_terminal_bg};
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
    .terminal-box {{
        background-color: var(--bg-terminal);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        overflow: hidden;
    }}
    .terminal-dots {{
        padding: 0.6rem 0.8rem;
        background-color: var(--bg-terminal);
        border-bottom: 1px solid var(--border-subtle);
    }}
    .terminal-dots span {{
        display: inline-block;
        width: 11px; height: 11px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .dot-red {{ background-color: #FF5F56; }}
    .dot-yellow {{ background-color: #FFBD2E; }}
    .dot-green {{ background-color: #27C93F; }}
    .terminal-box .stCode, .terminal-box pre {{
        border-radius: 0 !important;
        background-color: var(--bg-terminal) !important;
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
                st.session_state.conversations[INTERVIEW_MODE].append(
                    {"role": "assistant", "content": posed_question}
                )
                st.rerun()

    st.markdown('<div class="section-label">Chat History</div>', unsafe_allow_html=True)

    # Sohbet geçmişi TÜM modların sorularını listeler (hangi moda ait olduğu
    # ikonla belirtilir) -- bir moddan diğerine geçildiğinde önceki modun
    # konuşması ekrandan kalkar ama burada silinene kadar erişilebilir kalır.
    # Her kullanıcı sorusu + ona ait yanıt bir "kayıt" sayılır, her kaydın
    # yanında tek başına silinebilmesi için bir çöp kutusu butonu var.
    any_history = False
    for mode_name, icon in MODE_ICONS.items():
        conv = st.session_state.conversations[mode_name]
        user_indices = [i for i, m in enumerate(conv) if m["role"] == "user"]
        for idx in user_indices:
            any_history = True
            label = conv[idx]["content"]
            label = (label[:24] + "…") if len(label) > 24 else label
            col_label, col_delete = st.columns([5, 1])
            col_label.markdown(f'<div class="history-item">{icon} {label}</div>', unsafe_allow_html=True)
            if col_delete.button("✕", key=f"del_{mode_name}_{idx}"):
                # Bu soruyu ve (varsa) hemen ardından gelen yanıtı kaldır.
                end = idx + 2 if idx + 1 < len(conv) else idx + 1
                del st.session_state.conversations[mode_name][idx:end]
                st.rerun()

    if not any_history:
        st.caption("No conversations yet.")

    st.divider()
    chunk_count = count_chunks()
    if chunk_count > 0:
        st.caption(f"📚 Knowledge base ready — {chunk_count} chunks indexed.")
    else:
        st.caption("📚 Knowledge base not ready yet.")

# --- Ana alan: sohbet (sol) + terminal/kod paneli (sağ) ---
chat_col, terminal_col = st.columns([3, 2])

with chat_col:
    st.markdown(f'<div class="javabot-label">— JAVABOT</div>', unsafe_allow_html=True)

    if not current_messages:
        st.markdown("Hi. I'm here to help you with Java.")

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

        # Kenar çubuğundaki "Sohbet Geçmişi" listesi ve sağdaki terminal paneli
        # bu script çalışmasının BAŞINDA (mesaj eklenmeden önce) render edildiği
        # için, güncel halin görünmesi amacıyla sayfayı yeniden çalıştırıyoruz.
        st.rerun()

with terminal_col:
    dots_html = (
        '<div class="terminal-dots">'
        '<span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span>'
        "</div>"
    )
    st.markdown(f'<div class="terminal-box">{dots_html}', unsafe_allow_html=True)
    # Aktif modun konuşmasından türetilir -- mod değişince veya bir kayıt
    # silinince otomatik olarak doğru moda ait kodu (veya hiç yoksa
    # bekleme mesajını) gösterir.
    last_code = _last_code_in(current_messages)
    if last_code:
        st.code(last_code, language="java", line_numbers=True)
    else:
        st.code("// System ready.\n// Waiting for your question.", language="java", line_numbers=True)
    st.markdown("</div>", unsafe_allow_html=True)
