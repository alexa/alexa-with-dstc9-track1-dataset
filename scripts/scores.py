from dataset_walker import DatasetWalker

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge 

import re

import sys
import json
import argparse

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self._detection_tp = 0.0
        self._detection_fp = 0.0
        self._detection_tn = 0.0
        self._detection_fn = 0.0
        
        self._selection_mrr5 = 0.0
        self._selection_r1 = 0.0
        self._selection_r5 = 0.0

        self._generation_bleu1 = 0.0
        self._generation_bleu2 = 0.0
        self._generation_bleu3 = 0.0
        self._generation_bleu4 = 0.0
        self._generation_meteor = 0.0
        self._generation_rouge_1 = 0.0
        self._generation_rouge_2 = 0.0
        self._generation_rouge_l = 0.0

    def _match(self, ref_knowledge, pred_knowledge):
        result = []
        for pred in pred_knowledge:
            matched = False
            for ref in ref_knowledge:
                if pred['domain'] == ref['domain'] and pred['entity_id'] == ref['entity_id'] and pred['doc_id'] == ref['doc_id']:
                    matched = True
            result.append(matched)
        return result
        
    def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            idx = relevance.index(True)
            result = 1.0/(idx+1)
        else:
            result = 0.0

        return result

    def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5):
        relevance = self._match(ref_knowledge, hyp_knowledge)[:k]

        if True in relevance:
            result = 1.0
        else:
            result = 0.0

        return result

    def _normalize_text(self, text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result
    
    def _bleu(self, ref_response, hyp_response, n=4):
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

        weights = [1.0/n] * n
        
        score = sentence_bleu([ref_tokens], hyp_tokens, weights)

        return score

    def _meteor(self, ref_response, hyp_response):
        score = single_meteor_score(ref_response, hyp_response, self._normalize_text)

        return score

    def _rouge(self, ref_response, hyp_response, mode='l'):
        ref_response = self._normalize_text(ref_response)
        hyp_response = self._normalize_text(hyp_response)

        rouge = Rouge()

        if mode == 'l':
            score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-l']['f']
        elif mode == 1:
            score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-1']['f']
        elif mode == 2:
            score = rouge.get_scores(hyp_response, ref_response)[0]['rouge-2']['f']
        else:
            raise ValueError("unsupported mode: %s" % mode)

        return score

                    
    def update(self, ref_obj, hyp_obj):
        if ref_obj['target'] is True:
            if hyp_obj['target'] is True:
                self._detection_tp += 1
                
                self._selection_mrr5 += self._reciprocal_rank(ref_obj['knowledge'], hyp_obj['knowledge'], 5)
                self._selection_r1 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 1)
                self._selection_r5 += self._recall_at_k(ref_obj['knowledge'], hyp_obj['knowledge'], 5)

                self._generation_bleu1 += self._bleu(ref_obj['response'], hyp_obj['response'], 1)
                self._generation_bleu2 += self._bleu(ref_obj['response'], hyp_obj['response'], 2)
                self._generation_bleu3 += self._bleu(ref_obj['response'], hyp_obj['response'], 3)
                self._generation_bleu4 += self._bleu(ref_obj['response'], hyp_obj['response'], 4)
                self._generation_meteor += self._meteor(ref_obj['response'], hyp_obj['response'])
                self._generation_rouge_l += self._rouge(ref_obj['response'], hyp_obj['response'], 'l')
                self._generation_rouge_1 += self._rouge(ref_obj['response'], hyp_obj['response'], 1)
                self._generation_rouge_2 += self._rouge(ref_obj['response'], hyp_obj['response'], 2)                                                             
            else:
                self._detection_fn += 1
        else:
            if hyp_obj['target'] is True:
                self._detection_fp += 1
            else:
                self._detection_tn += 1

    def _compute(self, score_sum):
        if self._detection_tp + self._detection_fp > 0.0:
            score_p = score_sum/(self._detection_tp + self._detection_fp)
        else:
            score_p = 0.0

        if self._detection_tp + self._detection_fn > 0.0:
            score_r = score_sum/(self._detection_tp + self._detection_fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return (score_p, score_r, score_f)
        
    def scores(self):
        detection_p, detection_r, detection_f = self._compute(self._detection_tp)
        
        selection_mrr5_p, selection_mrr5_r, selection_mrr5_f = self._compute(self._selection_mrr5)
        selection_r1_p, selection_r1_r, selection_r1_f = self._compute(self._selection_r1)
        selection_r5_p, selection_r5_r, selection_r5_f = self._compute(self._selection_r5)

        generation_bleu1_p, generation_bleu1_r, generation_bleu1_f = self._compute(self._generation_bleu1)
        generation_bleu2_p, generation_bleu2_r, generation_bleu2_f = self._compute(self._generation_bleu2)
        generation_bleu3_p, generation_bleu3_r, generation_bleu3_f = self._compute(self._generation_bleu3)
        generation_bleu4_p, generation_bleu4_r, generation_bleu4_f = self._compute(self._generation_bleu4)
        generation_meteor_p, generation_meteor_r, generation_meteor_f = self._compute(self._generation_meteor)
        generation_rouge_l_p, generation_rouge_l_r, generation_rouge_l_f = self._compute(self._generation_rouge_l)
        generation_rouge_1_p, generation_rouge_1_r, generation_rouge_1_f = self._compute(self._generation_rouge_1)
        generation_rouge_2_p, generation_rouge_2_r, generation_rouge_2_f = self._compute(self._generation_rouge_2)

        scores = {
            'detection': {
                'prec': detection_p,
                'rec': detection_r,
                'f1': detection_f
            },
            'selection': {
                'mrr@5': selection_mrr5_f,
                'r@1': selection_r1_f,
                'r@5': selection_r5_f,
            },
            'generation': {
                'bleu-1': generation_bleu1_f,
                'bleu-2': generation_bleu2_f,
                'bleu-3': generation_bleu3_f,
                'bleu-4': generation_bleu4_f,
                'meteor': generation_meteor_f,
                'rouge_1': generation_rouge_1_f,
                'rouge_2': generation_rouge_2_f,
                'rouge_l': generation_rouge_l_f
            }
        }

        return scores
        
def main(argv):
    parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing scores')

    args = parser.parse_args()

    with open(args.outfile, 'r') as f:
        output = json.load(f)
    
    data = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=True)

    metric = Metric()

    for (instance, ref), pred in zip(data, output):
        metric.update(ref, pred)
        
    scores = metric.scores()

    with open(args.scorefile, 'w') as out:
        json.dump(scores, out, indent=2)
    

if __name__ =="__main__":
    main(sys.argv)        
