import pickle
import numpy as np
import itertools 


class Word2Vec():
	"""Store embeddings """

	def __init__(self, filepath):

		## Items from the whole polyglot dataset
		self.Words, self.Embeddings = pickle.load(open(filepath, 'rb'),encoding='latin-1')
		# Mappings for O(1) retrieval:
		self.Word2id = {word: idx for idx, word in enumerate(self.Words)}
		self.Id2word = {idx: word for idx, word in enumerate(self.Words)}
		self.embeddings_shape = self.Embeddings.shape[1]

		## Items from a subset of the words
		self.words= []
		self.wmbeddings = []
		self.word2id = {}
		self.id2word = {}



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
		print("{}/{} words from train set not in polyglot embedding".format(c,n))

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
			#print("Word {} is not included in polyglot".format(word))
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











