from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from _01_complaints._01_preprocessing import ProcessedComplaint
from _01_complaints._02_embeddings import BertEmbeddingService


@dataclass(frozen=True)
class DiscoveredTopic:
    topic_id: int
    label: str
    keywords: list[str]
    complaint_ids: list[str]
    size: int


class BertTopicDiscovery:
    """Discover complaint themes with BERTopic, falling back to keyword clusters."""

    def __init__(
        self,
        embedding_service: BertEmbeddingService | None = None,
        top_n_words: int = 6,
    ) -> None:
        self.embedding_service = embedding_service or BertEmbeddingService()
        self.top_n_words = top_n_words
        self._model = None

    def fit_transform(self, complaints: list[ProcessedComplaint]) -> list[DiscoveredTopic]:
        if not complaints:
            return []

        bertopic_topics = self._fit_transform_with_bertopic(complaints)
        if bertopic_topics is not None:
            return bertopic_topics
        return self._keyword_topics(complaints)

    def _fit_transform_with_bertopic(
        self,
        complaints: list[ProcessedComplaint],
    ) -> list[DiscoveredTopic] | None:
        try:
            from bertopic import BERTopic
        except ImportError:
            return None

        texts = [complaint.text for complaint in complaints]
        embeddings = self.embedding_service.encode(texts)
        self._model = BERTopic(top_n_words=self.top_n_words, calculate_probabilities=True)
        topic_ids, _ = self._model.fit_transform(texts, embeddings)

        grouped: dict[int, list[str]] = defaultdict(list)
        for complaint, topic_id in zip(complaints, topic_ids, strict=True):
            grouped[int(topic_id)].append(complaint.complaint_id)

        discovered = []
        for topic_id, complaint_ids in grouped.items():
            words = self._model.get_topic(topic_id) or []
            keywords = [word for word, _ in words[: self.top_n_words]]
            label = "_".join(keywords[:3]) if keywords else f"topic_{topic_id}"
            discovered.append(
                DiscoveredTopic(
                    topic_id=topic_id,
                    label=label,
                    keywords=keywords,
                    complaint_ids=complaint_ids,
                    size=len(complaint_ids),
                )
            )
        return sorted(discovered, key=lambda topic: topic.size, reverse=True)

    def _keyword_topics(self, complaints: list[ProcessedComplaint]) -> list[DiscoveredTopic]:
        buckets: dict[str, list[ProcessedComplaint]] = defaultdict(list)
        for complaint in complaints:
            keyword = self._dominant_keyword(complaint.tokens)
            buckets[keyword].append(complaint)

        topics = []
        for index, (label, grouped_complaints) in enumerate(sorted(buckets.items())):
            keyword_counts = Counter(
                token
                for complaint in grouped_complaints
                for token in complaint.tokens
                if token not in {"the", "and", "for", "with", "this", "that"}
            )
            topics.append(
                DiscoveredTopic(
                    topic_id=index,
                    label=label,
                    keywords=[word for word, _ in keyword_counts.most_common(self.top_n_words)],
                    complaint_ids=[complaint.complaint_id for complaint in grouped_complaints],
                    size=len(grouped_complaints),
                )
            )
        return sorted(topics, key=lambda topic: topic.size, reverse=True)

    def _dominant_keyword(self, tokens: list[str]) -> str:
        if not tokens:
            return "unknown"
        counts = Counter(token for token in tokens if len(token) > 3)
        if not counts:
            return tokens[0]
        return counts.most_common(1)[0][0]
