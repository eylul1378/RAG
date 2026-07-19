"""
Veri işleme (ingestion) hattı.

documents/ klasöründeki PDF ve TXT dosyalarını okur, pasaj düzeyinde (1-3
paragraflık) parçalara böler, her parçanın embedding vektörünü Microsoft
Foundry Local üzerinden hesaplar ve parça + vektörü SQLite'a kaydeder.
"""
import os

from PyPDF2 import PdfReader

from config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DOCUMENTS_DIR, EMBEDDING_BATCH_SIZE
from database import clear_chunks, count_chunks, init_db, insert_chunk
from foundry_client import embed_texts

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
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
