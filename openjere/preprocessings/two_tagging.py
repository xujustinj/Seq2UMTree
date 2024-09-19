#! -*- coding:utf-8 -*-

import json
from typing import Dict, List, Optional

from overrides import overrides

from openjere.preprocessings.abc_preprocessor import ABC_data_preprocessing


class Twotagging_preprocessing(ABC_data_preprocessing):
    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        instance = json.loads(line)
        text = instance["text"]

        if "spo_list" in instance:
            spo_list = instance["spo_list"]

            if not self._check_valid(text, spo_list):
                return None
            spo_list = [
                {
                    "predicate": spo["predicate"],
                    "object": spo["object"],
                    "subject": spo["subject"],
                }
                for spo in spo_list
            ]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

        result = {
            "text": text,
            "spo_list": spo_list,
        }
        return json.dumps(result, ensure_ascii=False)

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        return True

    # @overrides
    # def gen_vocab(self, min_freq: int):
    #     super(Chinese_twotagging_preprocessing, self).gen_vocab(
    #         min_freq, init_result={"<pad>": 0}
    #     )
