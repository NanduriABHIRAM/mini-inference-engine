from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.training.preprocess import (
    FEATURE_COLUMNS,
    InputSchemaError,
    fit_preprocessor,
    transform_features,
    validate_transaction_frame,
)


def frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [[float(row * 100 + column) for column in range(len(FEATURE_COLUMNS))] for row in range(rows)],
        columns=FEATURE_COLUMNS,
    )


def test_scaler_is_fit_only_from_training_rows() -> None:
    training = frame(3)
    held_out = frame(1).assign(**{FEATURE_COLUMNS[0]: 9_999.0})
    preprocessor = fit_preprocessor(training)

    transformed_training = transform_features(training, preprocessor)
    transformed_held_out = transform_features(held_out, preprocessor)

    np.testing.assert_allclose(transformed_training.mean(axis=0), 0.0, atol=1e-6)
    assert transformed_held_out[0, 0] > 100.0
    assert preprocessor["feature_columns"] == FEATURE_COLUMNS


def test_feature_order_is_restored_even_if_upload_is_reordered() -> None:
    original = frame(2)
    reversed_columns = original.loc[:, list(reversed(FEATURE_COLUMNS))]
    preprocessor = fit_preprocessor(original)

    transformed = transform_features(reversed_columns, preprocessor)
    expected = transform_features(original, preprocessor)

    np.testing.assert_allclose(transformed, expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.drop(columns=[FEATURE_COLUMNS[0]]),
        lambda data: data.assign(Unrelated=1.0),
        lambda data: data.assign(**{FEATURE_COLUMNS[1]: np.nan}),
        lambda data: data.assign(Class=0.5),
    ],
)
def test_invalid_transaction_schemas_are_rejected(mutate) -> None:
    with pytest.raises(InputSchemaError):
        validate_transaction_frame(mutate(frame()))
