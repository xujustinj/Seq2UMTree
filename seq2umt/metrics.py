class F1Triplet:
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1 = 2 * self.A / (self.B + self.C)
        p = self.A / self.B
        r = self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(
        self,
        predictions: list[list[dict[str, str]]],
        gold_labels: list[list[dict[str, str]]],
        get_seq=lambda dic: (dic["object"], dic["predicate"], dic["subject"]),
    ):
        for g, p in zip(gold_labels, predictions):
            g_set = set("_".join(get_seq(gg)) for gg in g)
            p_set = set("_".join(get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)
