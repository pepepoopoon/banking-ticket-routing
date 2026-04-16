import pandas as pd
import pytest

from banking_ticket_routing.demo_data import make_smoke_data
from banking_ticket_routing.stress import (
    downsample_intent,
    inject_label_noise,
    inject_token_noise,
)


def test_label_noise_is_deterministic_and_changes_only_labels() -> None:
    frame = make_smoke_data(samples_per_intent=20, random_state=5)
    first, changed = inject_label_noise(frame, rate=0.2, random_state=11)
    second, repeated_count = inject_label_noise(frame, rate=0.2, random_state=11)

    assert changed == repeated_count > 0
    pd.testing.assert_frame_equal(first, second)
    assert first["text"].equals(frame["text"])
    assert int((first["intent"] != frame["intent"]).sum()) == changed


def test_token_noise_keeps_rows_and_changes_text() -> None:
    frame = make_smoke_data()
    noisy, changed_tokens = inject_token_noise(frame, rate=0.5, random_state=17)

    assert changed_tokens > 0
    assert noisy["ticket_id"].equals(frame["ticket_id"])
    assert not noisy["text"].equals(frame["text"])


def test_downsample_intent_preserves_minimum_class_size() -> None:
    frame = make_smoke_data(samples_per_intent=20)
    reduced = downsample_intent(
        frame,
        intent="transfer_pending",
        keep_fraction=0.2,
        random_state=19,
    )

    counts = reduced["intent"].value_counts()
    assert counts["transfer_pending"] == 8
    assert counts.drop("transfer_pending").eq(20).all()


@pytest.mark.parametrize("rate", [-0.1, 0.9])
def test_token_noise_rejects_invalid_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="токенного шума"):
        inject_token_noise(make_smoke_data(), rate=rate, random_state=1)
