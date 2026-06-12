from _01_complaints import (
    BertComplaintClassifier,
    BertTopicDiscovery,
    ComplaintEvaluator,
    ComplaintKPIMapper,
    ComplaintMonitor,
    ComplaintPreprocessor,
)


def test_sample_complaint_pipeline_runs() -> None:
    sample = {
        "complaint_id": "cmp-001",
        "text": (
            "I was charged twice for my subscription, support ignored me, "
            "and I will cancel unless I get a refund."
        ),
        "channel": "email",
    }

    preprocessor = ComplaintPreprocessor()
    classifier = BertComplaintClassifier()
    topic_discovery = BertTopicDiscovery()
    evaluator = ComplaintEvaluator()
    monitor = ComplaintMonitor()
    kpi_mapper = ComplaintKPIMapper()

    processed = preprocessor.transform_many([sample])
    predictions = classifier.predict_many(
        [(complaint.complaint_id, complaint.text) for complaint in processed]
    )
    topics = topic_discovery.fit_transform(processed)
    report = evaluator.evaluate(predictions, topics)
    decision = monitor.assess(report)
    impacts = kpi_mapper.map_many(predictions)

    assert predictions[0].label in classifier.labels
    assert predictions[0].confidence > 0
    assert "financial_remediation" in predictions[0].triggers
    assert topics
    assert decision.status in {"healthy", "warning", "critical"}
    assert impacts[0].affected_kpis
