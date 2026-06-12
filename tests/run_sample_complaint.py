from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _01_complaints import (  # noqa: E402
    BertComplaintClassifier,
    BertTopicDiscovery,
    ComplaintEvaluator,
    ComplaintKPIMapper,
    ComplaintMonitor,
    ComplaintPreprocessor,
)


def main() -> None:
    sample = {
        "complaint_id": "cmp-001",
        "text": (
            "I was charged twice for my subscription, support ignored me for a week, "
            "and I will cancel unless I get a refund today."
        ),
        "channel": "email",
        "segment": "premium",
    }

    preprocessor = ComplaintPreprocessor()
    processed = preprocessor.transform_many([sample])

    classifier = BertComplaintClassifier()
    predictions = classifier.predict_many(
        [(complaint.complaint_id, complaint.text) for complaint in processed]
    )

    topics = BertTopicDiscovery().fit_transform(processed)
    report = ComplaintEvaluator().evaluate(predictions, topics)
    decision = ComplaintMonitor().assess(report)
    impacts = ComplaintKPIMapper().map_many(predictions)

    output = {
        "prediction": asdict(predictions[0]),
        "topics": [asdict(topic) for topic in topics],
        "evaluation": asdict(report),
        "monitoring": asdict(decision),
        "business_kpi_impact": [asdict(impact) for impact in impacts],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
