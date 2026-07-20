"""
Veri işleme (ingestion) hattı.

documents/ klasöründeki PDF ve TXT dosyalarını okur, pasaj düzeyinde (1-3
paragraflık) parçalara böler, her parçanın embedding vektörünü Microsoft
Foundry Local üzerinden hesaplar ve parça + vektörü SQLite'a kaydeder.
"""
import os
import re
from collections import Counter

from PyPDF2 import PdfReader

from config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DOCUMENTS_DIR, EMBEDDING_BATCH_SIZE
from database import clear_chunks, count_chunks, init_db, insert_chunk
from foundry_client import embed_texts

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Bazı PDF'lerde (özellikle LaTeX/Beamer ile üretilen slaytlarda) gömülü
# fontun karakter eşlemesi bozuk olduğundan, Türkçe aksanlı harfler PyPDF2
# tarafından "boşluklu işaret + temel harf" olarak çıkarılıyor
# (örn. "T¨ urkc¸e" yerine "T¨ u rkc¸e" değil ama "¨ u" -> "ü", "¸ s" -> "ş").
# Bu eşleme, işareti takip eden harfle birleştirip doğru Türkçe karakteri üretir.
_DIACRITIC_MAP = {
    ("¨", "u"): "ü", ("¨", "U"): "Ü",
    ("¨", "o"): "ö", ("¨", "O"): "Ö",
    ("¸", "c"): "ç", ("¸", "C"): "Ç",
    ("¸", "s"): "ş", ("¸", "S"): "Ş",
    ("˘", "g"): "ğ", ("˘", "G"): "Ğ",
    ("˙", "I"): "İ",
}
_DIACRITIC_PATTERN = re.compile("([¨¸˘˙]) ?([a-zA-Z])")

# Bazı harflerde (özellikle büyük harflerde, örn. "C ¸" -> "Ç") işaret,
# değiştirdiği harften SONRA değil ÖNCE geliyor. Bu yüzden önce "harf + işaret"
# (geriye dönük) eşleşmelerini, ardından kalan "işaret + harf" (ileriye dönük)
# eşleşmelerini çözüyoruz.
_BACKWARD_DIACRITIC_MAP = {
    ("C", "¸"): "Ç", ("c", "¸"): "ç",
    ("S", "¸"): "Ş", ("s", "¸"): "ş",
    ("U", "¨"): "Ü", ("u", "¨"): "ü",
    ("O", "¨"): "Ö", ("o", "¨"): "ö",
    ("G", "˘"): "Ğ", ("g", "˘"): "ğ",
    ("I", "˙"): "İ",
}
_BACKWARD_DIACRITIC_PATTERN = re.compile("([a-zA-Z]) ?([¨¸˘˙])")


def _fix_diacritics(text: str) -> str:
    def backward_repl(match: re.Match) -> str:
        letter, mark = match.group(1), match.group(2)
        return _BACKWARD_DIACRITIC_MAP.get((letter, mark), match.group(0))

    def forward_repl(match: re.Match) -> str:
        mark, letter = match.group(1), match.group(2)
        return _DIACRITIC_MAP.get((mark, letter), letter)

    text = _BACKWARD_DIACRITIC_PATTERN.sub(backward_repl, text)
    text = _DIACRITIC_PATTERN.sub(forward_repl, text)
    return text


_TURKISH_MONTHS = (
    "Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"
)
_MONTH_PATTERN = re.compile(_TURKISH_MONTHS)


def _normalize_line(line: str) -> str:
    # Üstbilgi/altbilgideki tarih (gün/ay/yıl) ve sayfa numarası dönem
    # boyunca değiştiğinden, karşılaştırma öncesi sayıları VE ay adlarını
    # da normalize ediyoruz; aksi halde her ders haftası ayrı bir "kalıp"
    # olarak sayılıp hiçbiri tekrar eşiğini geçemiyor.
    normalized = re.sub(r"\d+", "#", line.strip())
    return _MONTH_PATTERN.sub("#", normalized)


def _strip_repeated_boilerplate(page_texts: list[str], min_page_fraction: float = 0.2) -> list[str]:
    """Slaytlarda/sayfalarda her sayfada tekrar eden üstbilgi-altbilgi satırlarını
    (örn. "Doç. Dr. ... Algoritma ve Programlama II ... 19/40") tespit edip kaldırır.

    Bir satır, sayı ve ay adları yok sayıldığında sayfaların en az
    min_page_fraction kadarında aynı şekilde görünüyorsa "kalıp metin"
    (boilerplate) sayılır -- gerçek içerik bu kadar sık tekrar etmez.
    """
    page_lines = [text.split("\n") for text in page_texts]

    normalized_counts: Counter = Counter()
    for lines in page_lines:
        seen_this_page = set()
        for line in lines:
            if not line.strip():
                continue
            normalized = _normalize_line(line)
            if normalized not in seen_this_page:
                normalized_counts[normalized] += 1
                seen_this_page.add(normalized)

    threshold = max(3, int(len(page_texts) * min_page_fraction))
    boilerplate = {norm for norm, count in normalized_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in page_lines:
        kept = [line for line in lines if _normalize_line(line) not in boilerplate]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


# Slaytların İÇERİĞİ değil, dersle ilgili idari/kapanış slaytlarının
# BAŞLIĞI bu kalıplardan biriyle eşleşiyorsa o sayfanın tamamı atılır.
# Sadece slaydın İLK satırı (başlık) kontrol edilir -- "kaynak kod" veya
# "...sorular..." gibi ifadelerin gerçek teknik içeriğin İÇİNDE geçtiği
# sayfaları (örn. derleyici, boolean tip anlatımı) yanlışlıkla silmemek için.
_NOISE_TITLE_PATTERNS = [
    re.compile(r"^Teşekkürler\s*!?\s*$", re.IGNORECASE),
    re.compile(r"^Sorular(ınız)?\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^Soru ve Cevap", re.IGNORECASE),
    re.compile(r"^Sorular ve Cevaplar", re.IGNORECASE),
    re.compile(r"^Gelecek Hafta", re.IGNORECASE),
    re.compile(r"^Sonraki Hafta", re.IGNORECASE),
    re.compile(r"^Kaynaklar ve Araçlar", re.IGNORECASE),
    re.compile(r"^Kaynaklar\s*$", re.IGNORECASE),
    re.compile(r"^Öğrenme Kaynakları", re.IGNORECASE),
    re.compile(r"^Başarı İçin\s*Öneriler", re.IGNORECASE),
    re.compile(r"^Ödev ve Pratik Önerileri", re.IGNORECASE),
]


def _is_noise_slide(page_text: str) -> bool:
    for line in page_text.split("\n"):
        stripped = line.strip()
        if stripped:
            return any(pattern.match(stripped) for pattern in _NOISE_TITLE_PATTERNS)
    return False


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    raw_pages = [_fix_diacritics(page.extract_text() or "") for page in reader.pages]
    content_pages = [p for p in raw_pages if not _is_noise_slide(p)]
    cleaned_pages = _strip_repeated_boilerplate(content_pages)
    return "\n\n".join(cleaned_pages)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


_MD_HEADING_PATTERN = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BOLD_PATTERN = re.compile(r"(\*\*|__)(.*?)\1")


def extract_text_from_md(path: str) -> str:
    """GitHub tarzı Markdown dosyalarını (örn. mülakat soru bankaları) okur.
    Başlık işaretlerini (#) ve kalın vurgu işaretlerini (**/__) kaldırır ama
    kod bloklarına dokunmaz -- mülakat sorularındaki kod örnekleri retrieval
    için değerli bağlam sağlar."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = _MD_HEADING_PATTERN.sub("", text)
    text = _MD_BOLD_PATTERN.sub(r"\2", text)
    return text


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".md":
        return extract_text_from_md(path)
    if ext == ".txt":
        return extract_text_from_txt(path)
    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Metni paragraflara göre böler, ardından paragrafları chunk_size sınırına kadar
    gruplayarak pasaj düzeyinde (yaklaşık 1-3 paragraf) parçalar oluşturur.
    Bağlamın parçalar arasında kopmaması için küçük bir karakter örtüşmesi (overlap) eklenir.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip()
        else:
            current = f"{current}\n\n{para}".strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def run_ingestion() -> dict:
    """documents/ klasörünü baştan sona işler ve SQLite veritabanını yeniden doldurur.

    Returns:
        dict: {"files_processed": int, "chunks_created": int, "skipped": list[str]}
    """
    init_db()
    clear_chunks()

    if not os.path.isdir(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    files_processed = 0
    skipped: list[str] = []

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):
        path = os.path.join(DOCUMENTS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()

        if not os.path.isfile(path) or ext not in SUPPORTED_EXTENSIONS:
            if os.path.isfile(path):
                skipped.append(filename)
            continue

        text = extract_text(path)
        chunks = chunk_text(text)
        if not chunks:
            skipped.append(filename)
            continue

        # Foundry Local'a EMBEDDING_BATCH_SIZE'lık gruplar halinde istek
        # gönderiyoruz. Yüzlerce parçayı tek seferde göndermek, yerel
        # embedding servisinde isteğin iptal edilmesine yol açabiliyor.
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            embeddings = embed_texts(batch)
            for chunk_content, embedding in zip(batch, embeddings):
                insert_chunk(source=filename, content=chunk_content, embedding=embedding)

        files_processed += 1

    return {
        "files_processed": files_processed,
        "chunks_created": count_chunks(),
        "skipped": skipped,
    }


if __name__ == "__main__":
    result = run_ingestion()
    print(f"İşlenen dosya sayısı: {result['files_processed']}")
    print(f"Oluşturulan parça (chunk) sayısı: {result['chunks_created']}")
    if result["skipped"]:
        print(f"Atlanan dosyalar: {', '.join(result['skipped'])}")
