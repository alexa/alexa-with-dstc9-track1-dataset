# DSTC9 Track 1 Dataset

This directory contains the official dataset for [DSTC9 Track 1](../README.md).

If you want to publish experimental results with this dataset or use the baseline models, please cite the following article:
```
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
```

**NOTE**: This paper describes an earlier version of the data, which differs from the official challenge dataset in detail.


## Data

We are releasing the data divided into the following three subsets:

* Training set
  * [logs.json](train/logs.json): training instances
  * [labels.json](train/labels.json): ground-truths for training instances
* Validation set:
  * [logs.json](val/logs.json): validation instances
  * [labels.json](val/labels.json): ground-truths for training instances
* Test set
  * logs.json: test instances (to be released later)
  * labels.json: ground-truths for test instances (to be released later)

The ground-truth labels for Knowledge Selection task refer to the knowledge snippets in [knowledge.json](knowledge.json).

Participants will develop systems to take *logs.json* as an input and generates the outputs following the **same format** as *labels.json*.

## JSON Data Formats

### Log Objects

The *logs.json* file includes the list of instances each of which is a partial conversation from the beginning to the target user turn.
Each instance is a list of the following turn objects:

* speaker: the speaker of the turn (string: "U" for user turn/"S" for system turn)
* text: utterance text (string)

### Label Objects

The *labels.json* file provides the ground-truth labels and human responses for the final turn of each instance in *logs.json*.
It includes the list of the following objects in the same order as the input instances:

* target: whether the turn is knowledge-seeking or not (boolean: true/false)
* knowledge: [
  * domain: the domain identifier referring to a relevant knowledge snippet in *knowledge.json* (string)
  * entity\_id: the entity identifier referring to a relevant knowledge snippet in *knowledge.json* (integer for entity-specific knowledge or string "*" for domain-wide knowledge)
  * doc\_id: the document identifier referring to a relevant knowledge snippet in *knowledge.json* (integer)
  ]
* response: knowledge-grounded system response (string)

NOTE: *knowledge* and *response* exist only for the target instances with *true* for the *target* value.

### Knowledge Objects

The *knowledge.json* contains the unstructured knowledge sources to be selected and grounded in the tasks.
It includes the domain-wide or entity-specific knowledge snippets in the following format:

* domain\_id: domain identifier (string: "hotel", "restaurant", "train", "taxi", etc.)
  * entity\_id: entity identifier (integer or string: "*" for domain-wide knowledge)
      * name: entity name (string; only exists for entity-specific knowledge)
      * docs
          * doc\_id: document identifier (integer)
            * title: document title (string)
            * body: document body (string)



