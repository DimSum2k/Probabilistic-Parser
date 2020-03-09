import numpy as np 

class OOV(object):
	def __init__(self, w2v, ngram, n1=2, n2=2, l=0.2, damerau=False):
		self.w2v = w2v
		self.ngram = ngram
		self.vocabulary = ngram.vocab_
		self.n1 = n1
		self.n2 = n2
		self.l = l
		self.damerau = damerau


	def get_best_sentence(self, sentence):
		"""THIS IS AN APPROXIMATION"""

		listw = []

		# get nearest neighbours
		for i,word in enumerate(sentence):
			if word in self.vocabulary:
				listw.append([word])
			else:
				word_leven = self.w2v.most_similar_levenshtein(word,k=self.n1, damerau=self.damerau)
				word_cosine = self.w2v.most_similar_embeddings(word, k=self.n2)
				listw.append(word_leven + word_cosine)

		best_sentence = []
		for i, wl in enumerate(listw):
			
				if len(wl)==1:
					best_sentence.append(wl[0])
				else:
					if i==0:
						scores = [self.proba_interpolation(w) for w in wl]
					else:
						scores = [self.proba_interpolation(w,best_sentence[i-1]) for w in wl]

					best_sentence.append(wl[np.argmax(scores)])

		return best_sentence




	def proba_interpolation(self,w1,w0=None):
		if w0 is not None:
			return np.log(self.l*self.ngram.uni[w1] + (1-self.l)*self.ngram.graph[w0].get(w1,0))
		else:
			return np.log(self.l*self.ngram.uni[w1] + (1-self.l)*self.ngram.graph["*"].get(w1,0))