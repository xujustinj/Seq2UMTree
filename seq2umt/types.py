from typing import Literal


OptimizerName = Literal[
    "adam",
    "sgd",
]

ComponentName = Literal[
    "subject",
    "predicate",
    "object",
]

SplitName = Literal[
    "train",
    "validation",
    "test",
]
