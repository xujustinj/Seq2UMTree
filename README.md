# Seq2UMTree

> Ranran Haoran Zhang, Qianying Liu, Aysa Xuemo Fan, Heng Ji, Daojian Zeng, Fei Cheng, Daisuke Kawahara, and Sadao Kurohashi.
> 2020.
> [Minimize Exposure Bias of Seq2Seq Models in Joint Entity and Relation Extraction](https://aclanthology.org/2020.findings-emnlp.23).
> In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 236–246, Online.
> Association for Computational Linguistics.

This fork of [`WindChimeRan/OpenJERE`](https://github.com/WindChimeRan/OpenJERE) has been reduced to a minimal reproduction of the Seq2UMTree model introduced in the above paper.

The original repository contains the code for comparing Seq2UMTree against several baselines, but the abstractions involved make it a bit hard to understand the model on its own.
Here, we remove all code paths unrelated to Seq2UMTree and lightly refactor for readability, including clearer variable names and type annotations.

## Requirements

```sh
conda env create --file=environment.yml
```

## Data

### EWebNLG

> **TODO**

### NYT-CopyRE

```sh
raw_data/NYT-CopyRE/load.sh
```

### Re-DocRED

> **TODO**
