# DSTC9 Track 1 Evaluation Dataset

This directory contains the evaluation dataset for [DSTC9 Track 1](../README.md).

## Evaluation Data

DSTC9 Track 1 evaluation data includes the following three subsets:
* Subset #1: the test partition of the augmented MultiWOZ 2.1 collected by the same methods as the [training/validation datasets](../data/). 
* Subset #2: multi-domain human-human written conversations about touristic information for San Francisco.
* Subset #3: multi-domain human-human spoken conversations (with manual transcriptions) about touristic information for San Francisco.

We are releasing the following data and resources:
* [logs.json](test/logs.json): the test instances listed in a random order with no identifier of what subset each instance belongs to.
* [knowledge.json](knowledge.json): the knowledge candidates for all three subsets including 12,039 snippets for five domains and 668 entities in total, which is a super set of the [knowledge.json](../data/knowledge.json) for the training/validation set.
* [db.json](db.json): the domain DB entries for the Subset #2 and #3.

All the json formats are the same as the [training/validation resources](../data/README.md#json-data-formats).

## Participation

Each participating team will submit **up to 5** system outputs for the test instances in [logs.json](test/logs.json).

The system outputs must follow the **same format** as [labels.json](../data/README.md#label-objects) for the training/validation sets.
Before making your submission, please double check if every file is valid with no error from the following script:
``` shell
$ python scripts/check_results.py --dataset test --dataroot data_eval/ --outfile [YOUR_SYSTEM_OUTPUT_FILE]
Found no errors, output file is valid
```
Any invalid submission will be excluded from the official evaluation.

Once you're ready, please make your submission by completing the **[Submission Form](https://forms.gle/x5kyhxrM3fr4uEcf9)** by 11:59PM UTC-12 (anywhere on Earth), September 28, 2020.
