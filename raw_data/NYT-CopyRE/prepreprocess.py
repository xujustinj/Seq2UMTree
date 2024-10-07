import json
from typing import Any


def yield_line(source: str):
    with open(source, "r", encoding="utf-8") as s:
        for line in s:
            line = line.strip("\n")
            if not line:
                continue
            else:
                yield line


def process_line(line: str) -> str:
    unit: dict[str, Any] = json.loads(line)
    assert isinstance(unit, dict)
    text: str = unit["sentText"]
    assert isinstance(text, str)
    triples: list[dict[str, str]] = unit["relationMentions"]
    assert isinstance(triples, list)
    spo_list = [
        {
            "subject": t["em1Text"],
            "predicate": t["label"],
            "object": t["em2Text"],
        }
        for t in triples
    ]
    new_unit = {"text": text, "spo_list": spo_list}
    new_line = json.dumps(new_unit) + "\n"

    return new_line


def reformat(source: str, target: str):
    with open(target, "w", encoding="utf-8") as t:
        for line in yield_line(source):
            new_line = process_line(line)
            t.write(new_line)


if __name__ == "__main__":
    import argparse

    from util.type import assert_type

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    args = parser.parse_args()

    reformat(
        source=assert_type(args.source, str),
        target=assert_type(args.target, str),
    )
