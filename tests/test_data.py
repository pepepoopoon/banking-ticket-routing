import pandas as pd
import pytest

from banking_ticket_routing.data import stratified_split, validate_frame
from banking_ticket_routing.demo_data import make_smoke_data


def test_validation_rejects_duplicate_text() -> None:
    frame = make_smoke_data()
    frame.loc[1, "text"] = frame.loc[0, "text"].upper()

    with pytest.raises(ValueError, match="Повторяющиеся тексты"):
        validate_frame(frame)


def test_stratified_split_is_disjoint_and_deterministic() -> None:
    first = stratified_split(make_smoke_data(), random_state=11)
    second = stratified_split(make_smoke_data(), random_state=11)
    train, validation, test, manifest = first

    assert set(train["ticket_id"]).isdisjoint(validation["ticket_id"])
    assert set(train["ticket_id"]).isdisjoint(test["ticket_id"])
    assert set(validation["ticket_id"]).isdisjoint(test["ticket_id"])
    assert all(part["intent"].nunique() == 5 for part in (train, validation, test))
    pd.testing.assert_frame_equal(manifest, second[3])
