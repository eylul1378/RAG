"""
Veri işleme (ingestion) hattı.

documents/ klasöründeki PDF, TXT ve Markdown (.md) dosyalarını okur, pasaj
düzeyinde (1-3 paragraflık) parçalara böler, her parçanın embedding
vektörünü Microsoft Foundry Local üzerinden hesaplar ve parça + vektörü
SQLite'a kaydeder.
"""
import os
import re

from PyPDF2 import PdfReader

from config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DOCUMENTS_DIR, EMBEDDING_BATCH_SIZE
from database import clear_chunks, count_chunks, init_db, insert_chunk
from foundry_client import embed_texts

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


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


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Tek başına chunk_size'ı aşan bir paragrafı, mümkünse cümle sınırlarında,
    değilse (noktalama içermeyen uzun bir blok ise) ham karakter dilimleriyle böler.

    Bu olmadan chunk_text() tek uzun bir paragrafı hiç bölmeden olduğu gibi bir
    parçaya koyabiliyordu (gözlemlenen bir örnekte 1662 karakter, yapılandırılan
    800 sınırının iki katı) -- büyük bir parça, CPU'da üretim sırasında prefill
    süresini patlatıp neredeyse boş cevaplara yol açabiliyor.
    """
    if len(paragraph) <= chunk_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current.strip():
        pieces.append(current.strip())

    # Tek bir "cümle" bile chunk_size'ı aşıyorsa (örn. noktalama işareti
    # olmayan uzun bir kod bloğu), son çare olarak ham karakter dilimleme yap.
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            final_pieces.append(piece)
        else:
            for i in range(0, len(piece), chunk_size):
                final_pieces.append(piece[i : i + chunk_size])
    return final_pieces


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Metni paragraflara göre böler, ardından paragrafları chunk_size sınırına kadar
    gruplayarak pasaj düzeyinde (yaklaşık 1-3 paragraf) parçalar oluşturur.
    Bağlamın parçalar arasında kopmaması için küçük bir karakter örtüşmesi (overlap) eklenir.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    # Aşırı uzun paragrafları önceden böl ki aşağıdaki gruplama hiçbir zaman
    # chunk_size'ı ciddi şekilde aşan bir parça üretmesin.
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_long_paragraph(para, chunk_size))

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
