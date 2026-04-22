from cdc_priority.data.schema import DatasetSchema


def test_schema_feature_columns() -> None:
    schema = DatasetSchema(
        target="priority_label",
        categorical_columns=["event_type"],
        numeric_columns=["record_size"],
    )
    assert schema.feature_columns == ["event_type", "record_size"]
