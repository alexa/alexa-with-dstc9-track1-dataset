import numpy as np

from nltk import bigrams as get_bigrams
from nltk import trigrams as get_trigrams
from nltk import word_tokenize, ngrams
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from collections import Counter

from .data import normalize


def get_fourgrams(sequence, **kwargs):
    """
    Return the 4-grams generated from a sequence of items, as an iterator.

    :param sequence: the source data to be converted into 4-grams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 4, **kwargs):
        yield item


class Metric:
    def __init__(self):
        self.reset()
    
    def reset(self):
        pass
    
    def update(self, output):
        raise NotImplementedError()
    
    def compute(self):
        raise NotImplementedError()


class UnigramMetric(Metric):
    def __init__(self, metric):
        self._score = None
        self._count = None
        if metric.lower() not in ["recall", "precision"]:
            raise ValueError("mertic should be either 'recall' or 'precision', got %s" % metric)
        self.metric = metric.lower()
        super(UnigramMetric, self).__init__()

    def reset(self):
        self._score = 0
        self._count = 0
        super(UnigramMetric, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        hyp_tokens = normalize(hypothesis).split()
        ref_tokens = normalize(reference).split()

        common = Counter(ref_tokens) & Counter(hyp_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            score = 0
        else:
            if self.metric == "precision":
                score = 1.0 * num_same / len(hyp_tokens)
            else:
                assert self.metric == "recall"
                score = 1.0 * num_same / len(ref_tokens)

        self._score += score
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("Unigram metrics must have at least one example before it can be computed!")
        return self._score / self._count
    
    def name(self):
        return "Unigram{:s}".format(self.metric.capitalize())


class NGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n
        self._diversity = None
        self._count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("NGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")

        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(NGramDiversity, self).__init__()

    def reset(self):
        self._diversity = 0
        self._count = 0
        super(NGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output

        if hypothesis is None:
            diversity = 0
        else:
            diversity = 0
            output_tokens = word_tokenize(hypothesis)
            denominator = float(len(output_tokens))

            if denominator != 0.0:
                ngrams = set(list(self.ngram_func(output_tokens)))
                diversity = len(ngrams) / denominator

        self._diversity += diversity
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("NGramDiversity must consume at least one example before it can be computed!")
        return self._diversity / self._count

    def name(self):
        return "{:d}GramDiversity".format(self._n)


class CorpusNGramDiversity(Metric):
    def __init__(self, n=1):
        self._n = n

        self._ngrams = None
        self._token_count = None

        if self._n not in [1, 2, 3, 4]:
            raise ValueError("CorpusNGramDiversity only supports n=1 (unigrams), n=2 (bigrams),"
                             "n=3 (trigrams) and n=4 (4-grams)!")
        self.ngram_func = {
            1: lambda x: x,
            2: get_bigrams,
            3: get_trigrams,
            4: get_fourgrams
        }[self._n]

        super(CorpusNGramDiversity, self).__init__()

    def reset(self):
        self._ngrams = set()
        self._token_count = 0
        super(CorpusNGramDiversity, self).reset()

    def update(self, output):
        hypothesis, _ = output
        if isinstance(hypothesis, str) and hypothesis:
            output_tokens = word_tokenize(hypothesis)

            ngrams = list(self.ngram_func(output_tokens))
            self._ngrams.update(ngrams)
            self._token_count += len(output_tokens)

    def compute(self):
        if self._token_count == 0:
            raise ValueError("CorpusNGramDiversity must consume at least one example before it can be computed!")

        return len(self._ngrams) / self._token_count
    
    def name(self):
        return "Corpus{:d}GramDiversity".format(self._n)


class BLEU(Metric):
    def __init__(self):
        self._bleu = None
        self._count = None
        super(BLEU, self).__init__()

    def reset(self):
        self._bleu = 0
        self._count = 0
        super(BLEU, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        hyp_tokens = normalize(hypothesis).split()
        ref_tokens = normalize(reference).split()

        bleu = sentence_bleu([ref_tokens], hyp_tokens)

        self._bleu += bleu
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("BLEU-1 must have at least one example before it can be computed!")
        return self._bleu / self._count

    def name(self):
        return "BLEU"


class METEOR(Metric):
    def __init__(self):
        self._meteor = None
        self._count = None
        super(METEOR, self).__init__()

    def reset(self):
        self._meteor = 0
        self._count = 0
        super(METEOR, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        meteor = single_meteor_score(reference, hypothesis, preprocess=normalize)

        self._meteor += meteor
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("METEOR must have at least one example before it can be computed!")
        return self._meteor / self._count
    
    def name(self):
        return "METEOR"


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS

    This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge:
    """
    Class for computing ROUGE-L score for a set of candidate sentences

    This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
    with minor modifications
    """

    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate.split()

        for reference in refs:
            # split into tokens
            token_r = reference.split()
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def method(self):
        return "Rouge"


class ROUGE(Metric):
    def __init__(self):
        self.scorer = Rouge()
        self._rouge = None
        self._count = None
        super(ROUGE, self).__init__()

    def reset(self):
        self._rouge = 0
        self._count = 0
        super(ROUGE, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference = output

        rouge = self.scorer.calc_score(hypothesis, [reference])

        self._rouge += rouge
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("ROUGE-L must have at least one example before it can be computed!")
        return self._rouge / self._count

    def name(self):
        return "ROUGE"