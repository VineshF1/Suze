"""BM25 + Dense hybrid search with Reciprocal Rank Fusion."""

from typing import Optional

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import VectorStore


class HybridSearch:
    """Hybrid search combining BM25 sparse retrieval with ChromaDB dense retrieval via RRF."""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []
        self._corpus_meta: list[dict] = []

    def _build_bm25_index(self):
        """Build BM25 index from all stored chunks."""
        chunks = self.vs.get_all_chunks()
        self._corpus = [c["text"] for c in chunks]
        self._corpus_meta = [c["metadata"] for c in chunks]
        tokenized = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _ensure_index(self):
        if self._bm25 is None:
            self._build_bm25_index()

    def search_sparse(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 sparse retrieval."""
        self._ensure_index()
        if self._bm25 is None or not self._corpus:
            return []
        tokenized_query = query.split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append({
                    "text": self._corpus[idx],
                    "metadata": self._corpus_meta[idx],
                    "score": float(score),
                    "rank": len(results) + 1,
                })
        return results

    def search_dense(self, query: str, top_k: int = 20, where: Optional[dict] = None) -> list[dict]:
        """Dense vector search via ChromaDB."""
        result = self.vs.query(query_text=query, n_results=top_k, where=where)
        docs = []
        for i in range(len(result["ids"][0])):
            docs.append({
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": result["distances"][0][i] if result["distances"] else 0.0,
                "rank": i + 1,
            })
        return docs

    def hybrid_search(self, query: str, top_k: int = 10, sparse_k: int = 20,
                      dense_k: int = 20, where: Optional[dict] = None,
                      rrf_k: int = 60) -> list[dict]:
        """Hybrid search with Reciprocal Rank Fusion."""
        # When metadata pre-filters are active, skip sparse (BM25 can't filter metadata)
        if where:
            dense_results = self.search_dense(query, top_k=dense_k, where=where)
            # Rerank dense results by score
            ranked = sorted(dense_results, key=lambda x: x["score"])
            results = []
            for i, doc in enumerate(ranked):
                doc["rank"] = i + 1
                doc["score"] = 1.0 / (rrf_k + i)
                results.append(doc)
            return results[:top_k]

        sparse_results = self.search_sparse(query, top_k=sparse_k)
        dense_results = self.search_dense(query, top_k=dense_k, where=where)

        # RRF scoring
        rrf_scores: dict[int, float] = {}
        all_texts: list[str] = []

        for doc in sparse_results:
            text = doc["text"]
            idx = len(all_texts)
            all_texts.append(text)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + doc["rank"])

        for doc in dense_results:
            text = doc["text"]
            try:
                idx = all_texts.index(text)
            except ValueError:
                idx = len(all_texts)
                all_texts.append(text)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + doc["rank"])

        # Merge results
        text_to_meta = {}
        for doc in sparse_results + dense_results:
            if doc["text"] not in text_to_meta:
                text_to_meta[doc["text"]] = doc["metadata"]

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        results = []
        for idx, score in ranked[:top_k]:
            meta = text_to_meta.get(all_texts[idx], {})
            results.append({
                "text": all_texts[idx],
                "metadata": meta,
                "score": score,
                "rank": len(results) + 1,
            })
        return results
