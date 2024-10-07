import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from overrides import overrides

from openjere.config import ComponentName
from openjere.preprocessings.abc_preprocessor import AbstractPreprocessor
from util.find import find


class Seq2UMTreePreprocessor(AbstractPreprocessor):
    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        # for evaluation only
        instance = json.loads(line)
        assert isinstance(instance, dict)
        text = instance["text"]
        assert isinstance(text, str)

        spo_list = instance["spo_list"]
        assert isinstance(spo_list, list)
        if not self._check_valid(text, spo_list):
            return None
        extracted_spo_list: List[Dict[ComponentName, str]] = [
            {
                "predicate": spo["predicate"],
                "object": spo["object"],
                "subject": spo["subject"],
            }
            for spo in spo_list
        ]

        result = {
            "text": text,
            "spo_list": extracted_spo_list,
        }
        return json.dumps(result, ensure_ascii=False)

    def _train_read_line(self, line: str) -> Optional[List[str]]:
        # teacher forcing
        # batches are aligned by the triplets rather than sentences.
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

        result = self.spo_to_seq(text, spo_list)
        result = [json.dumps(r, ensure_ascii=False) for r in result]
        return result

    def spo_to_tree(
        self,
        spo_list: List[Dict[str, str]],
        order: Tuple[str, str, str] = ("predicate", "subject", "object")
    ) -> List[Tuple[str, str, List[str], List[str], List[str]]]:
        """return the ground truth of the tree: rel, subj, obj, used for teacher forcing.

        r: given text, one of the relations

        s: given r_1, one of the subjects

        rel: multi-label classification of relation

        subj: multi-label classification of subject

        obj: multi-label classification of object

        Arguments:
            spo_list {List[Dict[str, str]]} -- [description]

        Returns:
            List[Tuple[str]] -- [(r, s, rel, subj, obj)]
        """
        # rel, subj, obj
        result: List[Tuple[str, str, List[str], List[str], List[str]]] = []
        t1_out = list(set(t[order[0]] for t in spo_list))
        for t1_in in t1_out:
            t2_out = list(set(t[order[1]] for t in spo_list if t[order[0]] == t1_in))
            for t2_in in t2_out:
                t3_out = list(
                    set(
                        t[order[2]]
                        for t in spo_list
                        if t[order[0]] == t1_in and t[order[1]] == t2_in
                    )
                )
                result.append((t1_in, t2_in, t1_out, t2_out, t3_out))
        return result

    def spo_to_seq(self, text: str, spo_list: List[Dict[str, str]]):
        order = self.hyper.order

        tree = self.spo_to_tree(spo_list, order)
        tokens = self.hyper.tokenizer(text)

        def to_rel(outp: List[str]) -> List[Literal[0, 1]]:
            rel_idx: List[Literal[0, 1]] = [0] * len(self.relation_vocab)
            for rel_name in outp:
                rel_idx[self.relation_vocab[rel_name]] = 1
            return rel_idx

        def to_ent(outp: List[str]) -> Tuple[List[Literal[0, 1]], List[Literal[0, 1]]]:
            ent1: List[Literal[0, 1]] = [0] * len(tokens)
            ent2: List[Literal[0, 1]] = [0] * len(tokens)
            for name in outp:
                name_tokens = self.hyper.tokenizer(name)
                id = find(tokens, name_tokens)
                ent1[id] = 1
                ent2[id + len(name_tokens) - 1] = 1
            return ent1, ent2

        def to_in_key(inp: Optional[str], name: ComponentName) -> Union[int, Tuple[int, int]]:
            if not inp:
                return -1, -1
            elif name == "predicate":
                return self.relation_vocab[inp]
            else:
                inp_tokens = self.hyper.tokenizer(inp)
                k1 = find(tokens, inp_tokens)
                k2 = k1 + len(inp_tokens) - 1
                return k1, k2

        results: List[Dict[str, Any]] = []
        for t in tree:
            t1_in, t2_in, t1_out, t2_out, t3_out = t

            for name, ori_out, ori_in in zip(
                order, (t1_out, t2_out, t3_out), (t1_in, t2_in, None)
            ):
                if name == "predicate":
                    rel_idx = to_rel(ori_out)
                    rel_in = to_in_key(ori_in, name)
                elif name == "subject":
                    s1, s2 = to_ent(ori_out)
                    k = to_in_key(ori_in, name)
                    assert isinstance(k, tuple)
                    s_k1, s_k2 = k
                elif name == "object":
                    o1, o2 = to_ent(ori_out)
                    k = to_in_key(ori_in, name)
                    assert isinstance(k, tuple)
                    o_k1, o_k2 = k
                else:
                    raise ValueError("should be in predicate, subject, object")

            result = {
                "text": text,
                "spo_list": spo_list,
                "r": rel_in,
                "s_k1": s_k1,
                "s_k2": s_k2,
                "o_k1": o_k1,
                "o_k2": o_k2,
                "rel_gt": rel_idx,
                "s1_gt": s1,
                "s2_gt": s2,
                "o1_gt": o1,
                "o2_gt": o2,
            }

            results.append(result)
        return results

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if len(spo_list) == 0:
            return False

        for t in spo_list:
            if t["object"] not in text or t["subject"] not in text:
                return False
        return True

    @overrides
    def gen_all_data(self):
        print("Override gen_all_data: Different formats of train/dev!")

        for path in self.hyper.raw_data_list:
            if "train" in path:
                self.gen_train_data(path)
            else:
                self._gen_one_data(path)

    def gen_train_data(self, path):
        source = os.path.join(self.raw_data_root, path)
        target = os.path.join(self.data_root, path)
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line = line.strip("\n")
                if line == "":
                    continue
                newlines = self._train_read_line(line)
                if newlines is not None:
                    for newline in newlines:
                        t.write(newline)
                        t.write("\n")
