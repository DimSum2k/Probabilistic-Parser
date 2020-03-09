from pprint import pformat

class Automaton():
    """Dynamic graph to store Ngram transitions from a corpus of text.
    This representation of Ngrams is more memory efficient than using transition arrays
    and it scales better and generalises easily to Ngrams with N > 2.
    
    Parameters
    ----------
    N : int, default=2 -> START WITH ONLY BIGRAM
        N from Ngram, it will store P(w_N|w_1,...,w_{N-1})
        
    Attributes
    ----------
    N : from Ngram, it will store P(w_N|w_1,...,w_{N-1})
    
    graph : dictionary of dictionary representing the graph of transition.
        Each entry in the first level dictionary is a node
        Each node is a sequence of words with length < N
        Each vertex contains the probability of going from one node to another
            if node1 = (w1,...,w_{N-1}) and node2 = (w2,...,w_N),
            the vertex from node1 to node2 stores P(w_N|w_1,...,w_{N-1})
        
    vocab_ : stores the different words seen in the vocabulary when calling the fit method
    """
    
    def __init__(self, N=2):
        assert N>1 
        self.N = N
        self.graph = {"*":{}}
        self.uni = {}
        self.vocab_ = []
        

    def fit(self, corpus):
        '''Fill the graph with Ngram transitions from the corpus'''
        
        #flatten = lambda l: [item for sublist in l for item in sublist]
        #corpus = flatten(corpus)
        
        for sentence in corpus:
            if len(sentence)==0: continue
            node = "*"
            
            for i in range(len(sentence)):
    
                # create the node (a tuple of words)
                next_node = self.pad_node(sentence,i)
                # add 1 to the vertex going from the current node to the next node
                self.graph[node][next_node]  = self.graph[node].get(next_node, 0) + 1
                           
                node = next_node
                if node not in self.graph:
                    # create the node if it does not exist
                    self.graph[node] = {}
                    
                # update vocabulary
                #if sentence[i] not in self.vocab_:
                #    self.vocab_.append(sentence[i])

                # update count for unigram
                self.uni[sentence[i]] = self.uni.get(next_node, 0) + 1
            
            # "$" is the end of sentence symbol
            self.graph[node]["$$"] = self.graph[node].get("$$", 0) + 1
            
        # normalise to obtain for each node a distribution on the next nodes
        self.vocab_ = self.uni.keys()
        self.normalize()
        

    def pad_node(self,l,i):
        if self.N==2:
            return l[i]
        ll = l
        if i-self.N+2<0:
            ll = ["*"]*(self.N-2-i) + ll
            return "$".join(ll[:self.N-1])
        return "$".join(ll[i-self.N+2:i+1])
        


    def normalize(self):
        """For every node, normalise the outgoing vertices to obtain
        a probility ditribution on the next nodes"""
        
        for w in self.graph:
            self.graph[w] = {k: v / total for total in (sum(self.graph[w].values()),) for k, v in self.graph[w].items()}

        T = sum(self.uni.values())
        self.uni = {k:v/T for k,v in self.uni.items()}
                

    def __len__(self):
        return len(self.vocab_)
    
    def __getitem__(self, key):
        """Overload [] operator for handy access to transitio probability"""
        
        if type(key)==str:
            return self.graph["*"].get(key, 0)
        
        for i in range(2,self.N+1):
            if len(key)==i:
                key1 = tuple(key[:-1])
                key2 = tuple(key[1-self.N:])
                if key1 not in self.graph:
                    return 0
                else:
                    return self.graph[key1].get(key2, 0)
            
        raise ValueError("Unmatched key length")
        
    def __repr__(self):
        return pformat(self.graph, indent=1, width=1)