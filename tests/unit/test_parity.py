"""Training-serving parity: serving must accept exactly what training uses."""

from mta_abtest_engine.data.ingest import FEATURE_COLUMNS
from mta_abtest_engine.serving.schemas import Record


def test_api_schema_matches_training_features():
    assert list(Record.model_fields) == FEATURE_COLUMNS
