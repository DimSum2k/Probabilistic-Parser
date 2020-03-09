import pickle
import numpy as np
import itertools 


def sum_dic(dic1,dic2):
	dic = {}
	keys = set(dic1.keys()).intersection(dic2.keys())
	for k in keys:
		dic[k] = dic1.get(k,0) + dic2.get(k,0)
	return dic


def countOOV(sentence, pcfg):
	c = 0
	for w in sentence:
		if w not in pcfg.lexicon_:
			print("Word {} not in vocabulary".format(word))
			c+=1
	return c


def proba_interpolation(w1,l,ngram,w0=None):
	if w0 is not None:
		return np.log(l*ngram.uni[w1] + (1-l)*ngram.graph[w0].get(w1,0))
	else:
		return np.log(l*ngram.uni[w1] + (1-l)*ngram.graph["*"].get(w1,0))


def decode(listw,lists):
	# = list(itertools.permutations([len(s) for s in lists]))
	sentences = [element for element in itertools.product(*listw)]
	scores = []
	for sentence in sentences:
		s = 0
		for i,w in enumerate(sentence):
			if i==0: 
				s+= lists[0][listw.index[w]]


def extract_POS(sentence, lexicon):
	POS = []
	for w in sentence:
		dic = lexicon[w]
		POS.append(max(dic, key=dic.get))

	return POS



def get_best_sentence(sentence, vocabulary, w2v=None, ngram=None, n1=2, n2=2, l=0.2):
	"""FILL = improve since all words do not communicate ? """

	listw = []

	for i,word in enumerate(sentence):
		if word in vocabulary:
			listw.append([word])
		else:
			print("Word {} not in vocabulary".format(word))
			word_leven = w2v.most_similar_levenshtein(word,k=n1)
			word_cosine = w2v.most_similar_embeddings(word, k=n2)
			listw.append(word_leven + word_cosine)

	# all possible production of sentences
	sentences = [element for element in itertools.product(*listw)]

	best_sentence = []
	maxi_score = -np.infty
	for s in sentences:
		score = 0 
		for i,w in enumerate(s):
			if i==0: 
				score+= proba_interpolation(w,l,ngram)
			else: 
				score += proba_interpolation(w,l,ngram,s[i-1])

		if score>maxi_score:
			best_sentence = s
			maxi_score = score

	return best_sentence






"""
def get_POS_old(sentence, pcfg, w2v=None, ngram=None, n1=2, n2=2, l=0.2):

	listw = []
	lists = []

	for i,word in enumerate(sentence):

		if word in pcfg.lexicon_:

			#for j in range(len(words)):
			listw.append([word])

			if i==0:
				lists.append([proba_interpolation(word,l,ngram)])
			else:
				score = []
				for w0 in listw[i-1]:
					score.append(proba_interpolation(word,l,ngram,w0))
				lists.append(score)

		else:
			print("Word {} not in vocabulary".format(word))
			score = []
			word_leven = w2v.most_similar_levenshtein(word,k=n1)
			word_cosine = w2v.most_similar_embeddings(word, k=n2)
			candidates = word_leven + word_cosine
			listw.append(candidates)
			
			if i==0:
				for w in candidates:
					score.append(proba_interpolation(w,l,ngram))
				lists.append(score)

			else:
				for w1 in candidates:
					for w0 in listw[i-1]:
						score.append(proba_interpolation(w1,l,ngram,w0))
				lists.append(score)

	return decode(listw, lists)#decode(listw, lists)
"""


#n = len(sentence)
#c = countOOV(sentence, pcfg)
#scores = np.zeros((c*n1*n2,n))
#words = [[]]*(c*n1*n2)
#return [get_POS_w(w, pcfg, w2v, ngram,n1, n2) for w in sentence]
#print(word_leven, word_cosine)
#print(pcfg.lexicon_[word_leven])
#print(pcfg.lexicon_[word_cosine])
#dic = sum_dic(pcfg.lexicon_[word_leven],pcfg.lexicon_[word_cosine])
#return max(dic, key=dic.get)
#dic = pcfg.lexicon_[word]
#print(dic)
#return max(dic, key=dic.get)


class Word2Vec():

	def __init__(self, filepath):

		self.Words, self.Embeddings = pickle.load(open(filepath, 'rb'),encoding='latin-1')
		# Mappings for O(1) retrieval:
		self.Word2id = {word: idx for idx, word in enumerate(self.Words)}
		self.Id2word = {idx: word for idx, word in enumerate(self.Words)}
		self.embeddings_shape = self.Embeddings.shape[1]



	def extract_subset(self,words):

		n=len(words)
		words = list(words)
		embeddings = np.zeros((n, self.embeddings_shape))

		c = 0
		for i,w in enumerate(words):
			idx = self.Word2id[w] if w in self.Word2id else None
			if idx is not None:
				embeddings[i] = self.Embeddings[idx]
			else:
				c+=1
				#print("Word {} not in embedding list".format(w))       
		print("{}/{} words not in embedding list".format(c,n))

		self.words, self.embeddings = words, embeddings
		self.word2id = {word: idx for idx, word in enumerate(self.words)}
		self.id2word = {idx: word for idx, word in enumerate(self.words)}
		
	
	   
	def encode(self, word):
		"""Return embedding of a word from the vocabulary
		If the word is not present in the vocabulary, return an array of 0s
		
		Parameters
		----------
		word : str
			query word
			
		Returns
		-------
		embedding : array-like, shape (300,)
		"""
		
		idx = self.word2id.get(word, -1)
		if idx == -1:
			#print("Word {} is not included in the vocabulary".format(word))
			return np.zeros(self.embeddings.shape[1])
			
		return self.embeddings[idx]
		
			
	def score(self, word1, word2, encoded=False):
		"""Return cosine similarity between word1 and word2
		If encoded==True both words are assumed to be in encoded format
		If encoded==False both words are assumed to be in string format
		
		Parameters
		----------
		word1, word2 : str if encoded==False else array-like (shape, (300,))
			query words
		encoded: bool, default: False
			
		Returns
		-------
		score : float
			cosine similarity
		"""

		if encoded:
			e1,e2 = word1,word2
		else:
			e1,e2 = self.encode(word1),self.encode(word2)

		if np.sum(e1)==0 or np.sum(e2)==0: return 0
		
		# cosine similarity
		score = np.dot(e1, e2)  / (np.linalg.norm(e1)*np.linalg.norm(e2))
		
		return score    



	def levenshtein(self, w1,w2):
		#w1, w2 = w1.lower(), w2.lower() Bad for NP ? 
		n1, n2 = len(w1), len(w2)
		m = np.zeros((n1+1,n2+1))
		m[:,0] = range(n1+1)
		m[0,:] = range(n2+1)
		
		for i in range(1,n1+1):
			for j in range(1,n2+1):
				if w1[i-1]==w2[j-1]:
					m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1])
				else:
					m[i,j] = min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]+1)
					
		return m[n1,n2]   


	def damerau_levenshtein(self, s1,s2):

		#s1, s2 = s1.lower(), s2.lower() Bad for NP ? ????

		d = {}
		lenstr1 = len(s1)
		lenstr2 = len(s2)
		for i in range(-1,lenstr1+1):
			d[(i,-1)] = i+1
		for j in range(-1,lenstr2+1):
			d[(-1,j)] = j+1

		for i in range(lenstr1):
			for j in range(lenstr2):
				if s1[i] == s2[j]:
					cost = 0
				else:
					cost = 1
				d[(i,j)] = min(
							   d[(i-1,j)] + 1, # deletion
							   d[(i,j-1)] + 1, # insertion
							   d[(i-1,j-1)] + cost, # substitution
							  )
				if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
					d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

		return d[lenstr1-1,lenstr2-1]

	


	def most_similar_embeddings(self, word, k=1):
		"""Return k most similar words to 'word' in term of cosine similarity
		
		Parameters
		----------
		word : str 
			query word
		k : int, default: 5
			number of similar words
			
		Returns
		-------
		similar_words: list
		"""

		idx = self.Word2id.get(word, -1)
		if idx == -1:
			print("Word {} is not included in polyglot".format(word))
			return []
		word  = self.Embeddings[idx]

		scores = [self.score(word, w, encoded=True) for w in self.embeddings]
		closest_k = np.argsort(scores)[::-1][:k]
		
		return [self.id2word[i] for i in closest_k]



	def most_similar_levenshtein(self, word, k=1, damerau=False):
		"""Return k most similar words to 'word' in term of levenshtein distance

		Parameters
		----------
		word : str 
			query word
		k : int, default: 5
			number of similar words

		Returns
		-------
		similar_words: list
		"""

		if damerau:
			scores = [self.damerau_levenshtein(word,w) for w in self.words]
		else:
			scores = [self.levenshtein(word,w) for w in self.words]

		closest_k = np.argsort(scores)[:k]
		#print(np.sort(scores)[:k])

		return [self.words[i] for i in closest_k]



