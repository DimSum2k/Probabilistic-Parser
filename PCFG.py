import nltk

class PCFG(object):
	"""TO FILL"""


	def __init__(self):

		self.pcfg_ = {}
		self.lexicon_ = {}
		#self.terminal_ = []
		#self.axioms_ = {}
		self.POS = set()
		self.fitted_pcfg = False
		self.fitted_lexicon = False



	def fit_pcfg(self, X):
		"""Build PCFG from a list of consituency trees
			1- convert bracketed tree to nltk tree format
			2- apply Chomsky normal form transform
			3- extract rules and exclude the word leaves
			4- store transitions left->right 
			5- normalise to get probability ditributions
		
		Parameters
		----------
		X : list
			list of sentences in bracketed formats

		Returns
		-------
		pcfg : dic (inplace)
			Transition probabilities from symbol to symbol
		"""

		if self.fitted_pcfg:
			raise ValueError("PCFG.pcfg already fitted")

		for sentence in X:
			# nltk format
			t = nltk.tree.Tree.fromstring(sentence, remove_empty_top_bracketing=True)
			# chomky normal form
			self.chomkysation(t)
			#rules exraction
			rules = self.extract_rules(t, lexical=False)

			# fill transitions
			for r in rules:
				# if left side does not exist we create the node
				if r.lhs().symbol() not in self.pcfg_:
					self.pcfg_[r.lhs().symbol()] = {}
				# add one to vertex from left side to right side    
				rhs = (r.rhs()[0].symbol(), r.rhs()[1].symbol())
				self.pcfg_[r.lhs().symbol()][rhs] = self.pcfg_[r.lhs().symbol()].get(rhs,0) + 1
				
		self.pcfg_ = self.normalize(self.pcfg_)
		self.fitted_pcfg=True


	def fit_lexicon(self,X):
		"""Build lexicon from a list of consituency trees
			1- convert bracketed tree to nltk tree format -> find a way to to at the same time as pcfg
			[[[[2- apply Chomsky normal form transform]]]]] -> NO
			3- extract rules at the leaves
			4- store transitions right->left
			5- normalise to get probability ditributions
		
		Parameters
		----------
		X : list
			list of sentences in bracketed formats

		Returns
		-------
		lexicon : dic (inplace)
			Transition probabilities from words to terminal symbols (POS)
		"""

		# BUILD PCFG?  
	    #PCFG_grammar = defaultdict(set)
	    #for key, val in data[:,:2]:
	    #    if len(val)>1: PCFG_grammar[key.symbol()].add((val[0].symbol(),val[1].symbol()))
	    #    else: PCFG_grammar[key.symbol()].add(val[0].symbol())



		if self.fitted_lexicon:
			raise ValueError("PCFG.lexicon already fitted")

		for sentence in X:
			# nltk format
			t = nltk.tree.Tree.fromstring(sentence,remove_empty_top_bracketing=True)
			# chomky normal form
			self.chomkysation(t)
			rules = self.extract_rules(t, lexical=True)
			# rules extraction
			for r in rules:       
				# if right side does not exist we create the node
				rhs = r.rhs()[0]
				if rhs not in self.lexicon_:
					self.lexicon_[rhs] = {}
				# add one to vertex from right side (token) to left side (POS)   
				self.lexicon_[rhs][r.lhs().symbol()] = self.lexicon_[rhs].get(r.lhs().symbol(),0) + 1

				self.POS.add(r.lhs().symbol())
				
		self.POS = list(self.POS)
		self.lexicon_ = self.normalize(self.lexicon_)
		self.fitted_lexicon=True


	def chomkysation(self,t):
		"""Apply Chomsky normal form tranform inplace.
		
		Parameters
		----------
		t : nltk.tree.Tree
			Contituency tree in nklt format
		"""
		t.chomsky_normal_form(horzMarkov=2)
		t.collapse_unary(collapsePOS=True, collapseRoot=True)



	def extract_rules(self, t, lexical=True):
		"""Extract rules from a constituency tree.
		If lexical=True return rules from terminal symboles to words
		If lexical=False return  all rules except rules from terminal symboles to words
		
		Parameters
		----------
		t : nltk.tree.Tree
			Contituency tree in nklt format

		Returns
		-------
		rules : list
			list of rules
		"""

		rules = t.productions()
		
		if lexical:
			return [r for r in rules if r.is_lexical()]
		else:
			return [r for r in rules if r.is_nonlexical()]



	def normalize(self,dic):
		"""dic is assumed to be a dict of dict where the values 
		in the dictionaries at depth 2 are positive floats

		'normalize' normalises the positifve floats to get 
		probability distributions at each node

		Parameters
		----------
		dic : dict
			dict of dict

		Returns
		-------
		dic : dict
			normalised version of dict
		"""

		for el in dic:
			dic[el] = {k: v / total for total in (sum(dic[el].values()),) for k, v in dic[el].items()}

		return dic





