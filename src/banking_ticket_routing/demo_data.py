"""Детерминированный синтетический набор банковских обращений."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

EXAMPLES = {
    "card_arrival": [
        "Where is my new card",
        "When will the replacement card arrive",
        "My physical card has not arrived yet",
        "How long does card delivery take",
        "Can I track the card shipment",
        "The card delivery is late",
        "I am waiting for my bank card",
        "What is the status of my card delivery",
        "Has my new card been shipped",
        "Tell me the expected card arrival date",
    ],
    "cash_withdrawal": [
        "How can I withdraw cash",
        "Where can I take money from an ATM",
        "I need to make a cash withdrawal",
        "Which cash machines can I use",
        "Can I withdraw money abroad",
        "Help me get cash from my account",
        "What is the daily cash withdrawal process",
        "I want to use an ATM for cash",
        "Where do I find a supported cash machine",
        "Explain how ATM withdrawals work",
    ],
    "transfer_pending": [
        "My bank transfer is still pending",
        "Why has the transfer not completed",
        "The recipient is waiting for my transfer",
        "How long can a pending transfer take",
        "My money transfer is stuck",
        "The transfer status has not changed",
        "When will my pending payment transfer finish",
        "Please check the delayed bank transfer",
        "A transfer is showing as pending",
        "Why is my outgoing transfer delayed",
    ],
    "balance_not_updated_after_bank_transfer": [
        "My balance did not update after a bank transfer",
        "The incoming transfer is missing from my balance",
        "Why is transferred money not in my account balance",
        "My account total stayed the same after the transfer",
        "The bank transfer arrived but balance is unchanged",
        "Please refresh my balance after the incoming transfer",
        "Transferred funds are absent from the displayed balance",
        "My balance has not increased after a transfer",
        "I received a transfer but cannot see it in the balance",
        "Account balance is wrong following a bank transfer",
    ],
    "card_payment_wrong_exchange_rate": [
        "The exchange rate on my card purchase is wrong",
        "Why was my foreign card payment converted incorrectly",
        "I got a bad exchange rate for a card transaction",
        "Check the currency conversion on my card purchase",
        "The card payment used an unexpected conversion rate",
        "Foreign purchase exchange amount looks incorrect",
        "My card transaction has the wrong currency rate",
        "Explain this poor card payment exchange rate",
        "The conversion for my overseas card purchase is wrong",
        "I was charged the wrong exchange rate by card",
    ],
}

PREFIXES = (
    "Please help",
    "I need assistance",
    "Could you explain",
    "I have a question",
    "Please check",
    "Can support clarify",
    "I am contacting support because",
    "Could someone investigate",
)

SUFFIXES = (
    "today",
    "in the mobile app",
    "for my account",
    "before my next payment",
    "as soon as possible",
    "while I am travelling",
    "after the latest update",
    "from yesterday",
)


def make_smoke_data(*, samples_per_intent: int = 10, random_state: int = 42) -> pd.DataFrame:
    if samples_per_intent < 8:
        raise ValueError("Для split и калибровки нужно минимум 8 строк на intent")
    rng = random.Random(random_state)
    rows: list[dict[str, str]] = []
    ticket_number = 1
    for intent, texts in EXAMPLES.items():
        candidates = list(texts)
        augmented = [
            f"{prefix}: {text.lower()} {suffix}"
            for text in texts
            for prefix in PREFIXES
            for suffix in SUFFIXES
        ]
        rng.shuffle(augmented)
        candidates.extend(augmented)
        for text in candidates[:samples_per_intent]:
            rows.append(
                {
                    "ticket_id": f"ticket-{ticket_number:03d}",
                    "text": text,
                    "intent": intent,
                }
            )
            ticket_number += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-intent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_smoke_data(
        samples_per_intent=args.samples_per_intent,
        random_state=args.seed,
    ).to_csv(destination, index=False)


if __name__ == "__main__":
    main()
