[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# DSTC9 Track 1 - Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access

---

This repository contains the data, scripts and baseline codes for [DSTC9](https://dstc9.dstc.community/) Track 1.

This challenge track aims to support frictionless task-oriented conversations, where the dialogue flow does not break when users have requests that are out of the scope of APIs/DB but potentially are already available in external knowledge sources.
Track participants will develop dialogue systems to understand relevant domain knowledge, and generate system responses with the relevant selected knowledge.

**Organizers:** Seokhwan Kim, Mihail Eric, Behnam Hedayatnia, Karthik Gopalakrishnan, Yang Liu, Chao-Wei Huang, Dilek Hakkani-tur

## Important Links
* [Task Description Paper](https://arxiv.org/abs/2006.03533)
* [Track Proposal](https://drive.google.com/file/d/0Bx4CHsnRHDmJMXBNd0xGcmk5cE5OQ1FJWDM3NTY3dWZLR3E4/view?usp=sharing)
* [Challenge Registration](https://forms.gle/jdT79eBeySHVoa1QA)
* [Data Formats](data/README.md)
* [Baseline Details](baseline/README.md)

If you want to publish experimental results with this dataset or use the baseline models, please cite the following article:
```
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
```

**NOTE**: This paper reports the results with an earlier version of the dataset and the baseline models, which will differ from the baseline performances on the official challenge resources.

## Tasks

This challenge track decouples between turns that could be handled by the existing task-oriented conversational models with no extra knowledge and turns that require external knowledge resources to be answered by the dialogue system.
We focus on the turns that require knowledge access as the evaluation target in this track by the following three tasks:

| Task #1 | Knowledge-seeking Turn Detection                                                                                                      |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To decide whether to continue the existing scenario or trigger the knowledge access branch for a given utterance and dialogue history |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | Binary class (requires knowledge access or not)                                                                                       |

| Task #2 | Knowledge Selection                                                                                                                   |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To select proper knowledge sources from the domain knowledge-base given a dialogue state at each turn with knowledge access           |
| Input   | Current user utterance, Dialogue context, Knowledge snippets                                                                          |
| Output  | Ranked list of top-k knowledge candidates                                                                                             |

| Task #3 | Knowledge-grounded Response Generation                                                                                                |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal    | To take a triple of input utterance, dialog context, and the selected knowledge snippets and generate a system response               |
| Input   | Current user utterance, Dialogue context, and Selected knowledge snippets                                                             |
| Output  | Generated system response                                                                                                             |

Participants will develop systems to generate the outputs for each task.
They can leverage the annotations and the ground-truth responses available in the training and validation datasets.

In the test phase, participants will be given a set of unlabeled test instances.
And they will submit **up to 5** system outputs for **all three tasks**.

**NOTE**: For someone who are interested in only one or two of the tasks, we recommend to use our baseline system for the remaining tasks to complete the system outputs.


## Evaluation

Each submission will be evaluated in the following task-specific automated metrics first:

| Task                                   | Automated Metrics          |
|----------------------------------------|----------------------------|
| Knowledge-seeking Turn Detection       | Precision/Recall/F-measure |
| Knowledge Selection                    | Recall@1, Recall@5, MRR@5  |
| Knowledge-grounded Response Generation | BLEU, ROUGE, METEOR        |

To consider the dependencies between the tasks, the scores for knowledge selection and knowledge-grounded response generation are weighted by knowledge-seeking turn detection performances. Please find more details from [scores.py](scripts/scores.py).

The final ranking will be based on **human evaluation results** only for selected systems according to automated evaluation scores.
It will address the following aspects: grammatical/semantical correctness, naturalness, appropriateness, informativeness and relevance to given knowledge.

## Data

In this challenge track, participants will use an augmented version of [MultiWoz 2.1](https://github.com/budzianowski/multiwoz) which includes newly introduced knowledge-seeking turns.
All the ground-truth annotations for Knowledge-seeking Turn Detection and Knowledge Selection tasks as well as the agent's responses for Knowledge-grounded Response Generation task are available to develop the components on the [training](data/train) and [validation](data/val) sets.
In addition, relevant knowledge snippets for each domain or entity are also provided in [knowledge.json](data/knowledge.json).

In the test phase, participants will be evaluated on the results generated by their models for two data sets: one is the unlabeled test set of the augmented MultiWoz 2.1, and the other is a new set of unseen conversations which are collected from scratch also including turns that require knowledge access.
To evaluate the generalizability and the portability of each model, the unseen test set will be collected on different domains, entities and locales than MultiWoz.

Data and system output format details can be found from [data/README.md](data/README.md).

## Timeline

* Training data released: Jun 15, 2020 
* Test data released: Sep 21, 2020
* Entry submission deadline: Sep 28, 2020
* Objective evaluation completed: Oct 12, 2020
* Human evaluation completed: Oct 19, 2020

## Rules

* Participation is welcome from any team (academic, corporate, non profit, government).
* The identity of participants will NOT be published or made public. In written results, teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team label (e.g. team5), in publications or presentations, if they desire, but may not identify the identities of other teams.
* Participants are allowed to use any external datasets, resources or pre-trained models.

## Contact

### Join the DSTC mailing list to get the latest updates about DSTC9
* To join the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/join

* To post a message: send your message to list@dstc.community

* To leave the mailing list: visit https://groups.google.com/a/dstc.community/forum/#!forum/list/unsubscribe

### For specific enquiries about DSTC9 Track1

Please feel free to contact: seokhwk (at) amazon (dot) com
