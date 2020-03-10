from pprint import pprint, pformat
import progressbar
import multiprocessing as mp
from multiprocessing import Pool
import time
import pickle
import sys

import numpy as np

from config import get_arguments
from utils import *
from Word2Vec import Word2Vec
from Ngram import Automaton
from OOV import OOV
from PCFG import PCFG
from CYK import parsePCYK
from eval import evaluate_corpus


def multiprocess_func(func, args, n_jobs=-1):
	if n_jobs == -1:
		n_jobs = mp.cpu_count()
	with Pool(n_jobs) as p:
		#for i, _ in enumerate(p.starmap(func,args), 1):
		#	sys.stderr.write('\rdone {0:%}'.format(i/len(args)))
		res = p.starmap(func,args)
	return res


def main(opt):

	# read corpus
	print("Loading corpus ...")
	corpus = read(opt.path_data)
	# split corpus
	print("Splitting corpus ...")
	corpus_train, corpus_val, corpus_test = split(corpus,train=opt.train_frac, val=opt.val_frac, test=opt.test_frac)

	print("Train corpus length : {} ({:.0f}%)".format(len(corpus_train),len(corpus_train)/len(corpus)*100))
	print("Valid corpus length : {} ({:.0f}%)".format(len(corpus_val),len(corpus_val)/len(corpus)*100))
	print("Test corpus length : {} (Last {:.0f}%)".format(len(corpus_test),len(corpus_test)/len(corpus)*100),"\n")

	# extract sentences (leaves of trees) as list of list of words
	sentences_train, POS_train = extract_sentences(corpus_train)
	sentences_val, POS_val = extract_sentences(corpus_val)
	sentences_test, POS_test = extract_sentences(corpus_test)

	# extract vocabulary (individual words)
	print("Building vocabulary ...")
	vocabulary = build_vocabulary(sentences_train)
	print("Train vocabulary size: ", len(vocabulary))
	vocabulary_indices = {word:idx for idx,word in enumerate(vocabulary)}

	# w2v embeddings
	print("Building polyglot Word2Vec ...")
	w2v = Word2Vec(opt.path_embeddings)
	w2v.extract_subset(vocabulary)

	# bigram / unigram
	print("Building language model ...")
	ngram = Automaton(N=2)
	ngram.fit(sentences_train)

	# OOV 
	oov = OOV(w2v, ngram,opt.n1,opt.n2,opt.l,damerau=opt.damerau)

	# build pcfg and lexicon
	print("Building pcfg and lexicon")
	pcfg = PCFG()
	pcfg.fit(corpus_train)

	"""
	#extract some easy trees
	true_trees = []
	sentences = []
	for i,s in enumerate(sentences_test):
		if len(s)<5:
			true_trees.append(corpus_test[i])
			sentences.append(s)

	corpus_test = true_trees
	sentences_test = sentences
	

	"""
	print("Scanning through OOV sentences ...")
	new_sentences = []
	tic = time.time()
	for l in progressbar.progressbar(range(len(sentences_test))):
		new_sentences.append(oov.get_best_sentence(sentences_test[l]))
	print(time.time()-tic)


	pickle.dump(new_sentences, open("results/oov_test.pk", "wb"))
	#new_sentences = pickle.load(open("results/oov_test.pk","rb"))


	print("Parsing with PCYK ...")
	if opt.use_multiprocessing:
		print("... with multiprocessing ...")
		rep = [pcfg for i in range(len(sentences_test)) ]
		sentences_multi = list(zip(sentences_test, new_sentences, rep))
		tic = time.time()
		parses_results = multiprocess_func(parsePCYK, sentences_multi, n_jobs=opt.n_cpus)
		toc = time.time()
		print("Time elapsed :",(toc-tic)//60,"min",np.round((toc-tic)%60,2),"s")


	else:
		parses_results = []
		tic = time.time()
		for l in progressbar.progressbar(range(len(sentences))):
			parses_results.append(parsePCYK(sentences_test[l],new_sentences[l], pcfg))
		print(time.time()-tic)

	pickle.dump(parses_results, open( "results/parse_results.pk", "wb" ) )
	scores, nb_not_parsed = evaluate_corpus(parses_results, corpus_test)
	pickle.dump(scores, open( "results/scores.pk", "wb" ) )
	print("{} sentences not parsed with current grammar".format(nb_not_parsed))
	print(np.mean(scores))

	return 



if __name__=="__main__":

	parser = get_arguments()
	opt = parser.parse_args()

	main(opt)










