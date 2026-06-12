from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from _02_transcripts._01_preprocessing import Conversation
from _02_transcripts._02_embeddings import TranscriptEmbeddingService


@dataclass(frozen=True)
class DiscoveredIntent:
    intent_id: int
    label: str
    keywords: list[str]
    conversation_ids: list[str]
    size: int


class ConversationIntentDiscovery:
    """Discover emerging intents across conversations."""

    DISCOVERY_NOISE_TOKENS = {
        "again",
        "also",
        "and",
        "appreciate",
        "are",
        "bye",
        "can",
        "cool",
        "for",
        "good",
        "hello",
        "hey",
        "hi",
        "how",
        "just",
        "know",
        "like",
        "need",
        "ok",
        "okay",
        "please",
        "still",
        "thank",
        "thanks",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "you",
        "your",
    }

    def __init__(
        self,
        embedding_service: TranscriptEmbeddingService | None = None,
        top_n_words: int = 6,
    ) -> None:
        self.embedding_service = embedding_service or TranscriptEmbeddingService()
        self.top_n_words = top_n_words
        self._model = None

    def fit_transform(self, conversations: list[Conversation]) -> list[DiscoveredIntent]:
        if not conversations:
            return []
        bertopic_intents = self._fit_transform_with_bertopic(conversations)
        if bertopic_intents is not None:
            return bertopic_intents
        return self._keyword_intents(conversations)

    def _fit_transform_with_bertopic(
        self,
        conversations: list[Conversation],
    ) -> list[DiscoveredIntent] | None:
        try:
            from bertopic import BERTopic
        except ImportError:
            return None

        texts = [self._discovery_text(conversation) for conversation in conversations]
        embeddings = self.embedding_service.encode(texts)
        self._model = BERTopic(top_n_words=self.top_n_words, calculate_probabilities=True)
        intent_ids, _ = self._model.fit_transform(texts, embeddings)

        grouped: dict[int, list[str]] = defaultdict(list)
        for conversation, intent_id in zip(conversations, intent_ids, strict=True):
            grouped[int(intent_id)].append(conversation.conversation_id)

        discovered = []
        for intent_id, conversation_ids in grouped.items():
            words = self._model.get_topic(intent_id) or []
            keywords = [word for word, _ in words if word not in self.DISCOVERY_NOISE_TOKENS][
                : self.top_n_words
            ]
            discovered.append(
                DiscoveredIntent(
                    intent_id=intent_id,
                    label="_".join(keywords[:3]) if keywords else f"intent_{intent_id}",
                    keywords=keywords,
                    conversation_ids=conversation_ids,
                    size=len(conversation_ids),
                )
            )
        return sorted(discovered, key=lambda intent: intent.size, reverse=True)

    def _keyword_intents(self, conversations: list[Conversation]) -> list[DiscoveredIntent]:
        buckets: dict[str, list[Conversation]] = defaultdict(list)
        for conversation in conversations:
            tokens = self._intent_tokens(conversation)
            label = Counter(tokens).most_common(1)[0][0] if tokens else "unknown"
            buckets[label].append(conversation)

        intents = []
        for index, (label, grouped_conversations) in enumerate(sorted(buckets.items())):
            keyword_counts = Counter(
                token
                for conversation in grouped_conversations
                for token in self._intent_tokens(conversation)
            )
            intents.append(
                DiscoveredIntent(
                    intent_id=index,
                    label=label,
                    keywords=[word for word, _ in keyword_counts.most_common(self.top_n_words)],
                    conversation_ids=[
                        conversation.conversation_id for conversation in grouped_conversations
                    ],
                    size=len(grouped_conversations),
                )
            )
        return sorted(intents, key=lambda intent: intent.size, reverse=True)

    def _discovery_text(self, conversation: Conversation) -> str:
        return " ".join(self._intent_tokens(conversation))

    def _intent_tokens(self, conversation: Conversation) -> list[str]:
        return [
            token
            for turn in conversation.customer_turns
            for token in turn.tokens
            if token not in self.DISCOVERY_NOISE_TOKENS
        ]
