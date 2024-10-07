from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Iterable, Optional, TypeVar

from dataclasses_json import dataclass_json, DataClassJsonMixin, LetterCase
from tqdm import tqdm


@dataclass_json(letter_case=LetterCase.SNAKE) # type: ignore
@dataclass(frozen=True)
class OldEntityMention(DataClassJsonMixin):
    pos: tuple[int, int]
    type: str
    sent_id: int
    name: str
    global_pos: tuple[int, int]
    index: str


@dataclass_json
@dataclass(frozen=True)
class OldRelation(DataClassJsonMixin):
    r: str
    h: int
    t: int
    evidence: list[int]


@dataclass_json(letter_case=LetterCase.CAMEL) # type: ignore
@dataclass(frozen=True)
class OldSample(DataClassJsonMixin):
    title: str
    vertex_set: list[list[OldEntityMention]]
    labels: list[OldRelation]
    sents: list[list[str]]


@dataclass_json
@dataclass(frozen=True)
class NewRelation(DataClassJsonMixin):
    subject: str
    predicate: str
    object: str


@dataclass_json(letter_case=LetterCase.SNAKE) # type: ignore
@dataclass(frozen=True)
class NewSample(DataClassJsonMixin):
    text: str
    spo_list: list[NewRelation]


T = TypeVar("T")
def majority(it: Iterable[T]) -> T:
    counts = defaultdict(int)
    for x in it:
        counts[x] += 1
    return max(counts.keys(), key=lambda k: counts[k])

def same(it: Iterable[T]) -> T:
    it = iter(it)
    x = next(it)
    for y in it:
        assert y == x
    return x

with open("raw_data/redocred/rel_info.json") as f:
    id2predicate = json.load(f)

def old2new(old: OldSample) -> Optional[NewSample]:
    text = " ".join(" ".join(sent) for sent in old.sents)
    try:
        entities = [
            same(mention.name for mention in mentions)
            for mentions in old.vertex_set
        ]
    except AssertionError:
        return None
    relations = [
        NewRelation(
            subject=entities[relation.h],
            predicate=id2predicate[relation.r],
            object=entities[relation.t],
        )
        for relation in old.labels
    ]
    return NewSample(text=text, spo_list=relations)


io = (
    ("train", "train"),
    ("dev", "validate"),
    ("test", "test"),
)

for split_i, split_o in io:
    print("=" * 80)
    old_path = f"raw_data/redocred/data/{split_i}_revised.json"
    new_path = f"raw_data/redocred/new_{split_o}_data.jsonl"

    with open(old_path) as f:
        old_samples_raw = json.load(f)
    assert isinstance(old_samples_raw, list)

    old_samples = [
        OldSample.from_dict(d)
        for d in tqdm(old_samples_raw, desc=f"[{split_o}] reading old samples")
    ]
    del old_samples_raw
    print(f"[{split_o}] read {len(old_samples)} old samples from {old_path}")

    unfiltered_new_samples = [
        old2new(old)
        for old in tqdm(old_samples, desc=f"[{split_o}] converting samples")
    ]
    del old_samples
    new_samples = [new for new in unfiltered_new_samples if new is not None]
    print(f"[{split_o}] successfully converted {len(new_samples)}/{len(unfiltered_new_samples)} samples")
    del unfiltered_new_samples

    with open(new_path, "w") as f:
        f.writelines([
            (new.to_json() + "\n")
            for new in tqdm(new_samples, desc=f"[{split_o}] writing new samples")
        ])
        print(f"[{split_o}] wrote {len(new_samples)} new samples to {new_path}")
        del new_samples
