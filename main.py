from pprint import pprint, pformat

from utils import read, split, build_vocabulary, extract_sentences
from PCFG import PCFG
from OOV import OOV
from Word2Vec import Word2Vec
from Ngram import Automaton
from config import get_arguments

def main():
	path_data="sequoia-corpus+fct.mrg_strict"
	path_embeddings="polyglot-fr.pkl"
	train_frac=0.9
	val_frac=0.0
	test_frac=0.1

	# read corpus
	corpus = read(path_data)
	# split corpus
	corpus_train, corpus_val, corpus_test = split(corpus,train=train_frac, val=val_frac, test=test_frac)
	print("Train corpus length : {} ({:.0f}%)".format(len(corpus_train),len(corpus_train)/len(corpus)*100))
	print("Valid corpus length : {} ({:.0f}%)".format(len(corpus_val),len(corpus_val)/len(corpus)*100))
	print("Test corpus length : {} (Last {:.0f}%)".format(len(corpus_test),len(corpus_test)/len(corpus)*100),"\n")

	# extract sentences (leaves of trees) as list of list of words
	sentences_train, POS_train = extract_sentences(corpus_train)
	sentences_val, POS_val = extract_sentences(corpus_val)
	sentences_test, POS_test = extract_sentences(corpus_test)

	# extract vocabulary (individual words)
	vocabulary = build_vocabulary(sentences_train)
	print("Train vocabulary size: ", len(vocabulary))
	vocabulary_indices = {word:idx for idx,word in enumerate(vocabulary)}

	# w2v embeddings
	w2v = Word2Vec(path_embeddings)
	print(len(w2v.Words),w2v.Embeddings.shape)
	w2v.extract_subset(vocabulary)
	print(len(w2v.words),w2v.embeddings.shape)

	# bigram / unigram
	ngram = Automaton(N=2)
	ngram.fit(sentences_train)

	# OOV 
	oov = OOV(w2v, ngram,5,5,0.1)




	return [corpus_train, corpus_val, corpus_test, sentences_train, POS_train, sentences_val, POS_val, sentences_test, POS_test, vocabulary, w2v, ngram, oov]



if __name__=="__main__":
	
	#parser = get_arguments()
	#opt = parser.parse_args()
	pass







