"""Three-stage hybrid filtering engine.

Stage 1 (Pre-filter): Hard metadata filters (date ranges, access levels, doc_type)
Stage 2 (ANN Vector Search): Semantic ranking on pre-filtered subset
Stage 3 (Post-filter): Lightweight refinement (author verification, tags)
"""

from datetime import datetime
from typing import Optional


class StagedFilter:
    """Three-stage filtering pipeline for production-grade retrieval."""

    def __init__(self, hybrid_search):
        self.hybrid_search = hybrid_search

    def retrieve(
        self,
        query: str,
        *,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        access_level: Optional[str] = None,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        author: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Full 3-stage retrieval pipeline."""
        # Stage 1: Build pre-filter
        pre_filter = self._build_pre_filter(
            date_from=date_from,
            date_to=date_to,
            access_level=access_level,
            doc_type=doc_type,
            source=source,
        )

        # Stage 2: ANN Vector Search (with metadata pre-filter)
        dense_k = top_k * 4  # Retrieve more for post-filtering
        results = self.hybrid_search.hybrid_search(
            query=query,
            top_k=dense_k,
            where=pre_filter if pre_filter else None,
        )

        # Stage 3: Post-filter refinement
        results = self._post_filter(results, tags=tags, author=author)

        return results[:top_k]

    def _build_pre_filter(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        access_level: Optional[str] = None,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[dict]:
        """Build ChromaDB $and/$or filter from metadata criteria.
        
        Dates are compared as Unix timestamps (float) — ChromaDB only supports
        numeric comparisons with $gte/$lte.
        """
        from datetime import datetime, timezone
        conditions = []

        if date_from or date_to:
            date_filter = {}
            if date_from:
                # Parse ISO string to Unix timestamp
                dt = datetime.fromisoformat(date_from)
                date_filter["$gte"] = dt.timestamp()
            if date_to:
                dt = datetime.fromisoformat(date_to)
                date_filter["$lte"] = dt.timestamp()
            conditions.append({"date_created": date_filter})

        if access_level:
            conditions.append({"access_level": access_level})

        if doc_type:
            conditions.append({"doc_type": doc_type})

        if source:
            conditions.append({"source": source})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _post_filter(
        self,
        results: list[dict],
        tags: Optional[list[str]] = None,
        author: Optional[str] = None,
    ) -> list[dict]:
        """Lightweight in-memory post-filtering."""
        if not tags and not author:
            return results

        filtered = []
        for doc in results:
            meta = doc.get("metadata", {})

            if tags:
                doc_tags = meta.get("tags", [])
                if isinstance(doc_tags, str):
                    doc_tags = [doc_tags]
                if not any(t in doc_tags for t in tags):
                    continue

            if author:
                doc_author = meta.get("author", "")
                if doc_author != author:
                    continue

            filtered.append(doc)

        return filtered
