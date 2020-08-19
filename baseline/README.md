[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Neural Baseline Models for DSTC9 Track 1

This directory contains the official baseline codes for [DSTC9 Track 1](../README.md).

It includes the neural baseline models described in the [Task Description Paper](https://arxiv.org/abs/2006.03533).

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

## Getting started

* Clone this repository into your working directory.

``` shell
$ git clone https://github.com/alexa/alexa-with-dstc9-track1-dataset.git
$ cd alexa-with-dstc9-track1-dataset
```

* Install the required python packages.

``` shell
$ pip3 install -r requirements.txt
$ python -m nltk.downloader 'punkt'
$ python -m nltk.downloader 'wordnet
```

* Train the baseline models.

``` shell
$ ./train_baseline.sh
```

* Run the baseline models.

``` shell
$ ./run_baseline.sh
```

* Validate the structure and contents of the tracker output.

``` shell
$ python scripts/check_results.py --dataset val --dataroot data/ --outfile baseline_val.json
Found no errors, output file is valid
```

* Evaluate the output.

``` shell
$ python scripts/scores.py --dataset val --dataroot data/ --outfile baseline_val.json --scorefile baseline_val.score.json
```

* Print out the scores.

``` shell
$ cat baseline_val.score.json | jq
{
  "detection": {
    "prec": 0.9992307692307693,
    "rec": 0.9719416386083053,
    "f1": 0.9853973070358429
  },
  "selection": {
    "mrr@5": 0.784069789493646,
    "r@1": 0.6728617485302485,
    "r@5": 0.929641570263607
  },
  "generation": {
    "bleu-1": 0.3600525451854145,
    "bleu-2": 0.2201514928053953,
    "bleu-3": 0.13890208351903832,
    "bleu-4": 0.0956298903190914,
    "meteor": 0.35999761796852026,
    "rouge_1": 0.39392136452739857,
    "rouge_2": 0.1748996273676227,
    "rouge_l": 0.35013871624013504
  }
}
```

