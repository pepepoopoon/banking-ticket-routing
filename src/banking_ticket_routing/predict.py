"""CLI маршрутизации одного обращения."""

from __future__ import annotations

import argparse
import json

from .service import RoutingService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    result = RoutingService.from_path(args.model).route(args.text, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
