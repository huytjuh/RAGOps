from _02_transcripts import (
    ConversationIntentDiscovery,
    ConversationKPIMapper,
    ConversationMonitor,
    QueryResponseJudge,
    TranscriptEvaluator,
    TranscriptIntentRecognizer,
    TranscriptPreprocessor,
    TurnSentimentAnalyzer,
)


def test_sample_transcript_pipeline_runs() -> None:
    transcript = """
    Customer: I still cannot log in and this is the third time I have contacted you.
    Agent: I can help reset your password and check the account lock.
    Customer: I am frustrated and may cancel if this keeps happening.
    Agent: I reset the account and sent a secure login link.
    """

    conversation = TranscriptPreprocessor().transform("conv-001", transcript)
    recognizer = TranscriptIntentRecognizer()
    turn_predictions = recognizer.predict_customer_turns(conversation)
    conversation_prediction = recognizer.predict_conversation(conversation)
    predictions = [*turn_predictions, conversation_prediction]
    sentiments = TurnSentimentAnalyzer().analyze_conversation(conversation)
    discovered = ConversationIntentDiscovery().fit_transform([conversation])
    judge = QueryResponseJudge().evaluate(
        query="Can the customer log in and what should happen next?",
        response="The agent reset the account and sent a secure login link.",
        evidence=[turn.raw_text for turn in conversation.turns],
        required_points=["reset account", "secure login link"],
    )
    report = TranscriptEvaluator().evaluate(predictions, discovered, sentiments, judge=judge)
    decision = ConversationMonitor().assess(report)
    impact = ConversationKPIMapper().map_prediction(conversation_prediction, sentiments)

    assert conversation.customer_turns
    assert all(prediction.intent in recognizer.intents for prediction in predictions)
    assert sentiments[0].sentiment in {"negative", "neutral", "positive"}
    assert discovered
    assert judge.overall > 0
    assert decision.status in {"healthy", "warning", "critical"}
    assert impact.affected_kpis


def test_customer_noise_turns_are_filtered() -> None:
    transcript = """
    Customer: Hi
    Agent: Hello, how can I help?
    Customer: thanks
    Customer: Hi, I cannot log in again.
    Customer: okay
    """

    conversation = TranscriptPreprocessor().transform("conv-noise", transcript)

    assert [turn.raw_text for turn in conversation.customer_turns] == ["Hi, I cannot log in again."]
    assert len(conversation.turns) == 2


def test_intent_discovery_filters_greeting_keywords() -> None:
    transcript = """
    Customer: Thanks, but I need a refund for the duplicate charge.
    Agent: I can review the payment.
    Customer: Hello, the invoice fee is still wrong.
    """

    conversation = TranscriptPreprocessor().transform("conv-discovery-noise", transcript)
    discovered = ConversationIntentDiscovery().fit_transform([conversation])

    assert discovered[0].label not in {"hello", "thanks", "need", "still"}
    assert "thanks" not in discovered[0].keywords
    assert "refund" in discovered[0].keywords or "invoice" in discovered[0].keywords
