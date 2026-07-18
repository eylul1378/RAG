"""SQLite işlemleri: belge parçalarını (chunk) ve embedding vektörlerini saklama/okuma."""
import json
import os
import sqlite3

import numpy as np

from config import DB_PATH, TOP_K
from foundry_client import embed_query


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """chunks tablosunu yoksa oluşturur. embedding, JSON-serileştirilmiş bir vektör olarak tutulur."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def clear_chunks() -> None:
    """Yeniden işleme (re-ingest) öncesi tabloyu boşaltır."""
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(source: str, content: str, embedding: list[float]) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def get_all_chunks() -> list[dict]:
    """Kosinüs benzerliği hesaplamak için tüm parçaları ve vektörlerini belleğe okur.

    Not: Bu, küçük ölçekli (birkaç yüz-bin parça) bir doküman koleksiyonu için
    yeterlidir. Çok daha büyük koleksiyonlarda özel bir vektör veritabanı gerekir.
    """
    conn = get_connection()
    rows = conn.execute("SELECT id, source, content, embedding FROM chunks").fetchall()
    conn.close()
    return [
        {"id": r[0], "source": r[1], "content": r[2], "embedding": json.loads(r[3])}
        for r in rows
    ]


def count_chunks() -> int:
    conn = get_connection()
    (count,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    conn.close()
    return count


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_top_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """Sorguyu Foundry Local ile vektörleştirir, SQLite'taki tüm vektörleri çekip
    kosinüs benzerliğini Python'da (numpy ile) yerel olarak hesaplar ve en
    alakalı top_k parçayı döndürür.

    Küçük ölçekli bir doküman koleksiyonu için brute-force karşılaştırma
    (tüm vektörleri belleğe okuyup tek tek karşılaştırma) yeterlidir; daha
    büyük koleksiyonlarda özel bir vektör veritabanı/indeks gerekir.
    """
    all_chunks = get_all_chunks()
    if not all_chunks:
        return []

    query_vector = np.array(embed_query(query))

    scored = []
    for chunk in all_chunks:
        similarity = _cosine_similarity(query_vector, np.array(chunk["embedding"]))
        scored.append((similarity, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _similarity, chunk in scored[:top_k]]
