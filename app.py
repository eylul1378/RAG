"""
JavaBot - Streamlit arayüzü.

Tamamen çevrimdışı çalışan, İngilizce kaynaklarla (Oracle Java dokümantasyonu,
Think Java kitabı, GitHub mülakat soru bankaları) beslenen ama HER ZAMAN
TÜRKÇE yanıt veren bir Java eğitmeni. Embedding ve sohbet modelleri Microsoft
Foundry Local üzerinden bu makinede çalıştırılır, hiçbir soru veya belge
içeriği internete gönderilmez.

Arayüz üç panelden oluşur (Figma tasarımına göre):
- Sol: mod seçimi (her modun kendi vurgu rengi vardır) ve silinebilir sohbet
  geçmişi.
- Orta: sohbet akışı ve "> ... ÇALIŞTIR" giriş kutusu.
- Sağ: terminal görünümlü bir kod paneli -- en son yanıt kod içeriyorsa
  otomatik olarak orada gösterilir, içermiyorsa bekleme mesajı görünür.

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
    "Bebek Adımları": "🏆",
    "Akademik Mod": "🎓",
    "Mülakat Senaryosu": "⚔️",
}
MODE_COLORS = {
    "Bebek Adımları": "#4ADE80",
    "Akademik Mod": "#F97316",
    "Mülakat Senaryosu": "#F43F5E",
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


# Sağ paneldeki "terminal" kutusu için: yanıt bir Java kod bloğu içeriyorsa
# onu ayıklayıp orada gösteriyoruz (kullanıcının istediği davranış budur --
# terminal paneli sadece dekoratif değil, kodun gittiği yer).
_CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    match = _CODE_BLOCK_PATTERN.search(text)
    if match:
        code = match.group(1).strip()
        return code or None
    return None


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
current_messages = st.session_state.conversations[active_mode]

# --- Koyu, terminal/kod-editörü esintili tema (Figma tasarımına göre) ---
st.markdown(
    f"""
    <style>
    :root {{
        --bg-main: #0B1220;
        --bg-panel: #0A0E17;
        --bg-terminal: #05070C;
        --border-subtle: #1E293B;
        --text-dim: #64748B;
        --active-color: {active_color};
    }}
    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace;
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
        background-color: #0A0E17;
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
    st.markdown('<div class="section-label">Çalışma Modu</div>', unsafe_allow_html=True)

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
                        posed_question = _strip_leaked_answer(posed_question)
                    except Exception as exc:
                        posed_question = f"Bir hata oluştu: {exc}"
                        chunk = None

                st.session_state.interview_chunk = chunk
                st.session_state.conversations[INTERVIEW_MODE].append(
                    {"role": "assistant", "content": posed_question}
                )
                st.rerun()

    st.markdown('<div class="section-label">Sohbet Geçmişi</div>', unsafe_allow_html=True)

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
            if col_delete.button("🗑", key=f"del_{mode_name}_{idx}"):
                # Bu soruyu ve (varsa) hemen ardından gelen yanıtı kaldır.
                end = idx + 2 if idx + 1 < len(conv) else idx + 1
                del st.session_state.conversations[mode_name][idx:end]
                st.rerun()

    if not any_history:
        st.caption("Henüz sohbet yok.")

    st.divider()
    chunk_count = count_chunks()
    if chunk_count > 0:
        st.caption(f"📚 Bilgi tabanı hazır — {chunk_count} belge parçası indekslendi.")
    else:
        st.caption("📚 Bilgi tabanı henüz hazırlanmadı.")

# --- Ana alan: sohbet (sol) + terminal/kod paneli (sağ) ---
chat_col, terminal_col = st.columns([3, 2])

with chat_col:
    st.markdown(f'<div class="javabot-label">— JAVABOT</div>', unsafe_allow_html=True)

    if not current_messages:
        st.markdown("Merhaba. Ben Java konularında yardımcı olmak için buradayım.")

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    placeholder = "Mülakat sorusuna cevabını yaz..." if active_mode == INTERVIEW_MODE else "Java'ya bir şeyler sorun..."
    question = st.chat_input(placeholder)

    if question:
        current_messages.append({"role": "user", "content": question})
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

            elif active_mode == INTERVIEW_MODE:
                if st.session_state.interview_chunk is None:
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
                    st.session_state.interview_chunk = None
                    st.caption("Yeni bir soru için kenar çubuğundaki '🎯 Yeni Mülakat Sorusu' butonuna tıkla.")

            else:
                top_chunks = get_top_chunks(question)

                if not top_chunks:
                    # Küçük modeller "bağlam yok, uydurma" talimatına rağmen kendi
                    # genel bilgisinden cevap uydurabiliyor (test edildi, güvenilmez).
                    # Bu yüzden LLM'i hiç çağırmadan -- hem daha hızlı hem %100
                    # garantili -- doğrudan Python'da "bulamadım" cevabını veriyoruz.
                    answer = "Bu bilgiyi belgelerde bulamadım."
                    st.markdown(answer)
                else:
                    with st.spinner("İlgili kaynak parçaları aranıyor ve yanıt hazırlanıyor..."):
                        try:
                            context_texts = [
                                f"[Kaynak: {chunk['source']}]\n{chunk['content']}" for chunk in top_chunks
                            ]
                            system_prompt = build_system_prompt(active_mode)
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
        st.code("// Sistem hazır.\n// Sorunuzu bekliyorum.", language="java", line_numbers=True)
    st.markdown("</div>", unsafe_allow_html=True)
