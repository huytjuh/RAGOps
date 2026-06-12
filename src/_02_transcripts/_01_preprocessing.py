from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationTurn:
    conversation_id: str
    turn_id: int
    speaker: str
    raw_text: str
    text: str
    tokens: list[str]


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    turns: list[ConversationTurn]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def customer_turns(self) -> list[ConversationTurn]:
        return [turn for turn in self.turns if turn.speaker == "customer"]

    @property
    def text(self) -> str:
        return " ".join(turn.text for turn in self.turns)


class TranscriptPreprocessor:
    """Split transcripts into normalized conversation turns."""

    CUSTOMER_NOISE_TOKENS = {
        "afternoon",
        "appreciate",
        "bye",
        "cool",
        "evening",
        "good",
        "goodbye",
        "hello",
        "hey",
        "hi",
        "morning",
        "ok",
        "okay",
        "please",
        "thank",
        "thanks",
        "thankyou",
        "welcome",
        "yes",
        "yep",
        "you",
    }
    CUSTOMER_NOISE_PHRASES = {
        "good afternoon",
        "good evening",
        "good morning",
        "hello",
        "hey",
        "hi",
        "how are you",
        "ok",
        "okay",
        "thank you",
        "thanks",
    }
    SPEAKER_PATTERN = re.compile(
        r"^\s*(customer|agent|user|assistant|rep|advisor)\s*:\s*(.+)$",
        re.I,
    )
    SPEAKER_MAP = {
        "advisor": "agent",
        "assistant": "agent",
        "rep": "agent",
        "user": "customer",
    }

    def __init__(self, filter_customer_noise: bool = True) -> None:
        self.filter_customer_noise = filter_customer_noise

    def transform(
        self,
        conversation_id: str,
        transcript: str,
        metadata: dict[str, str] | None = None,
    ) -> Conversation:
        turns = self.split_turns(conversation_id, transcript)
        return Conversation(conversation_id=conversation_id, turns=turns, metadata=metadata or {})

    def split_turns(self, conversation_id: str, transcript: str) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        pending_speaker = "customer"
        pending_text: list[str] = []
        saw_speaker_label = False

        def flush() -> None:
            if not pending_text:
                return
            raw_text = " ".join(pending_text).strip()
            normalized = self.normalize(raw_text)
            tokens = self.tokenize(normalized)
            is_noise = self.is_customer_noise(pending_speaker, normalized, tokens)
            if self.filter_customer_noise and is_noise:
                return
            turns.append(
                ConversationTurn(
                    conversation_id=conversation_id,
                    turn_id=len(turns),
                    speaker=pending_speaker,
                    raw_text=raw_text,
                    text=normalized,
                    tokens=tokens,
                )
            )

        for line in transcript.splitlines():
            line = line.strip()
            if not line:
                continue
            match = self.SPEAKER_PATTERN.match(line)
            if match:
                saw_speaker_label = True
                flush()
                speaker = match.group(1).lower()
                pending_speaker = self.SPEAKER_MAP.get(speaker, speaker)
                pending_text = [match.group(2)]
            else:
                pending_text.append(line)
        flush()

        if not turns and transcript.strip() and not saw_speaker_label:
            normalized = self.normalize(transcript)
            turns.append(
                ConversationTurn(
                    conversation_id=conversation_id,
                    turn_id=0,
                    speaker="customer",
                    raw_text=transcript.strip(),
                    text=normalized,
                    tokens=self.tokenize(normalized),
                )
            )
        return turns

    def normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " email_address ", text)
        text = re.sub(r"\b\d{4,}\b", " number ", text)
        text = re.sub(r"[^a-z0-9_\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        return [token for token in text.split() if len(token) > 1]

    def is_customer_noise(self, speaker: str, text: str, tokens: list[str]) -> bool:
        if speaker != "customer":
            return False
        if not tokens:
            return True
        if text in self.CUSTOMER_NOISE_PHRASES:
            return True

        substantive_tokens = [token for token in tokens if token not in self.CUSTOMER_NOISE_TOKENS]
        return not substantive_tokens
