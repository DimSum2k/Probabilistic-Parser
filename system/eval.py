from PYEVALB import parser as evalbparser
from PYEVALB import scorer
import numpy as np


def evaluate_corpus(corpus_pred, corpus_ref):

	score = np.zeros(len(corpus_pred))
	nb_not_parsed = 0
	for i, (sentence, reference) in enumerate(zip(corpus_pred, corpus_ref)):
		if get_not_parsed(sentence):
			nb_not_parsed += 1
			score[i] = -np.infty
			continue

		score[i] = evaluate(sentence,reference)

	return score, nb_not_parsed


def evaluate(sentence,reference):
    gold_tree = evalbparser.create_from_bracket_string(sentence[1:-1])
    test_tree = evalbparser.create_from_bracket_string(reference[1:-1])

    s = scorer.Scorer()
    result = s.score_trees(gold_tree, test_tree)
    
    return result.tag_accracy


def get_not_parsed(s):
	c=0
	for l in s:
		if l=="(" or l==")":
			c+=1
	if c==4:
		return True
	return False 



