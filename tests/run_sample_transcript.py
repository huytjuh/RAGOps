from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _02_transcripts import (  # noqa: E402
    ConversationIntentDiscovery,
    ConversationKPIMapper,
    ConversationMonitor,
    QueryResponseJudge,
    TranscriptEvaluator,
    TranscriptIntentRecognizer,
    TranscriptPreprocessor,
    TurnSentimentAnalyzer,
)


def main() -> None:
    transcript = """
    Customer: I still cannot log in and this is the third time I have contacted support.
    Agent: I can help reset your password and check the account lock.
    Customer: I am frustrated and may cancel if this keeps happening.
    Agent: I reset the account and sent a secure login link.
    Customer: Thanks, but I need to know it will not break again.
    """

    conversation = TranscriptPreprocessor().transform(
        "conv-001",
        transcript,
        metadata={"channel": "chat", "segment": "premium"},
    )
    recognizer = TranscriptIntentRecognizer()
    turn_predictions = recognizer.predict_customer_turns(conversation)
    conversation_prediction = recognizer.predict_conversation(conversation)
    predictions = [*turn_predictions, conversation_prediction]

    sentiments = TurnSentimentAnalyzer().analyze_conversation(conversation)
    discovered = ConversationIntentDiscovery().fit_transform([conversation])
    judge = QueryResponseJudge().evaluate(
        query="What happened and what should the assistant tell the customer?",
        response="The agent reset the account and sent a secure login link.",
        evidence=[turn.raw_text for turn in conversation.turns],
        required_points=["account reset", "secure login link", "login issue"],
    )
    report = TranscriptEvaluator().evaluate(predictions, discovered, sentiments, judge=judge)
    decision = ConversationMonitor().assess(report)
    kpi_mapper = ConversationKPIMapper()
    impact = kpi_mapper.map_prediction(conversation_prediction, sentiments)

    output = {
        "turns": [asdict(turn) for turn in conversation.turns],
        "turn_intents": [asdict(prediction) for prediction in turn_predictions],
        "conversation_intent": asdict(conversation_prediction),
        "sentiment": [asdict(sentiment) for sentiment in sentiments],
        "discovered_intents": [asdict(intent) for intent in discovered],
        "judge": asdict(judge),
        "evaluation": asdict(report),
        "monitoring": asdict(decision),
        "business_kpi_impact": asdict(impact),
        "deployment_kpi_summary": kpi_mapper.deployment_kpi_summary(decision),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
