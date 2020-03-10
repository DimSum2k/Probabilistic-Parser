#!/bin/bash -e

# Create script as "script.py"
cat >script.py <<'END_SCRIPT'

import argparse
from collections import Counter
import operator
import copy
import progressbar
import numpy as np
import pickle
from PYEVALB import scorer
from PYEVALB import parser

parser = argparse.ArgumentParser(description='Process arguments of sh')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('-paths','--paths', nargs='+', help='list of paths tree bank and embeddings', required=True)
#parser.add_argument('-sentence','--sentence', action='store_true', default=False, help="True if you want to proceed a sentence")
parser.add_argument('-files','--files', nargs='+', help='list of string, sentence and "NA" if sentence, filepath and "None" or target path', required=True)
parser.add_argument('--sentence','--sentence', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="True if you want to proceed a sentence")
parser, _ = parser.parse_known_args()







class CorpusManager:
    """grammar class"""

    #polyglot-fr.pkl
    def __init__(self,corpus):
        self.corpus = corpus
        self.grammar = {}
        self.lexic = {}
        self.unigram = {} 
        self.bigram ={}
        self.sentences=[]
        self.dic_nodes={}  
        self.term_node=[]
        self.mat_grammar=[]

        
    
    ###################Tools########################    
   
    
    #Erase the part after the dash in POS
    def erase_dash(self):
        for i,sent in enumerate(self.corpus):
            temp = sent.split('(')
            for j,k in enumerate(temp):
                if not len(k.split())==0:
                    temp[j] =  k.split()[0].split('-')[0]+' '+' '.join(k.split()[1:])
            self.corpus[i]='('.join(temp)

                  
    #Return  a list of words from a bracket sentence
    def bracket_to_word(self,sent):
        sent = sent[7:]   
        words=[]
        tmp=sent.split(")")
        for t in tmp:
            try:
                words.append(t.split()[-1])
            except:
                pass
        return words
     
    #build lexic
    def build_lexic(self):
        lexic={}

        for sent in self.corpus:
            st=sent.split("(")
            for i in range(len(st)):
                if ")" in st[i]:
                    l=st[i].replace(")","")
                    word=l.split()[1]

                    tp=l.split()[0]
                    tp=tp.split("-")[0]
                    if word not in lexic.keys():
                        lexic[word] = []
                    lexic[word].append(tp)

        for key in lexic.keys():
                count=Counter(lexic[key])
                lexic[key]=dict(count)

        for key in lexic.keys():
            count=lexic[key]
            total = float(sum(count.values()))
            for k in count:
                count[k] /= total
        self.lexic=lexic
    
    #build bigram values
    def build_bigram(self):
        
        bigrams={}

        for u,sentence in enumerate(self.corpus):
            sentence=self.bracket_to_word(sentence)
            sentence=np.append(sentence,"#e")

            # "#s" and "#e" indicate the starting state and the end state
            sentence=np.insert(sentence,0,"#s")
            for i in range(len(sentence)-1):
                seq = sentence[i]
            
                if  seq not in bigrams.keys():
                    bigrams[seq] = []
                next_chain=sentence[i+1]
                
                if "#e" in next_chain:
                    bigrams[seq].append("#e")
                else:
                    bigrams[seq].append(next_chain)
    
        #count occurences
        for key in bigrams.keys():
            bigrams[key]=Counter(bigrams[key])

        for key in bigrams.keys():
            count=bigrams[key]
            total = float(sum(count.values()))
            for k in count:
                count[k] /= total
            bigrams[key]=dict(count)
        self.bigram = bigrams
         
    # build unigram values
    def build_unigram(self):
        dic_uni={}
        for sent in self.corpus:
            s=self.bracket_to_word(sent)
            for w in s:
                if w not in dic_uni.keys():
                    dic_uni[w]=1
                else:
                    dic_uni[w]+=1
        
       
        total = float(sum(dic_uni.values()))
        for k,v in dic_uni.items():
            dic_uni[k] /= total
        self.unigram = dic_uni
            
    #in order to evaluate on the test
    def build_sentences(self):
        for s in self.corpus:
            self.sentences.append(self.bracket_to_word(s))
    
    #create a dictionary of indices in order to make the CYK algorithm faster
    def build_nodes(self):
        list_nodes=[]
        for k in self.grammar.keys():
            list_nodes.append(k)
            for kb in self.grammar[k].keys():
                r=kb.split("_")
                list_nodes.append(r[0])
                if len(r)==2:
                    list_nodes.append(r[1])
                    
        list_nodes=list(set(list_nodes))
        self.dic_nodes={n:i for i,n in enumerate(list_nodes)}
          

    ###################Build PCFG########################
    
    #build a tree from a bracket sentence
    def build_child(self,sentence):
        dico={}
        sentence_c=sentence[1:-1]
        name_node=sentence_c.split()[0]
        dico[name_node]={}
        u=0
        terminal=(sentence_c.find("(")==-1)
        if(not terminal):
            sentence_c=sentence_c[sentence_c.find("("):]

            while(len(sentence_c.strip())>0):
                i=0
                find_open=False
                while find_open==False:
                    if sentence_c[i]=="(":
                        find_open=True
                        start=i
                        score=0
                        j=i
                        find_close_one=False
                        while find_close_one==False:
                            if sentence_c[j]=="(":
                                score+=-1
                            if sentence_c[j]==")":
                                score+=1
                            if score==0:
                                end=j
                                find_close_one=True
                            j+=1
                    i+=1

                element_to_remove="("+sentence_c[start:(end+1)][1:-1]+")"
                sentence_c=sentence_c[0:start]+sentence_c[(end+1):]
                key=str(u)
                u=u+1
                dico[name_node][name_node+"_"+key]=self.build_child(element_to_remove)

        else:

            v=sentence.split()
            sentence=" ".join(v) 
            dico=sentence 
            return(dico)

        return(dico)

    #add rules on a grammar from the previous tree
    def find_child(self,dico,grammar_tree):

        for k in dico.keys():
            node=k
        if node not in grammar_tree.keys():
            grammar_tree[node]=[]

        list_son=[]
        for k, value in dico[node].items():

            if type(value) == dict:
                for kb in value.keys():
                    son=kb
                self.find_child(dico[node][k],grammar_tree)
            else:
                son=value.split()[0][1:]

            list_son.append(son)

        grammar_tree[node].append("_".join(list_son))
       
    #build pcfg, add all rules to grammar and normalize to obtain probabilities
    def build_pcfg(self):

        for sentence in self.corpus:

            sentence=sentence[2:(len(sentence)-1)]
            self.find_child(self.build_child(sentence),self.grammar)

        for k,v in self.grammar.items():
            self.grammar[k]=dict(Counter(v))


        for k in self.grammar.keys():
            count=self.grammar[k]
            total = float(sum(count.values()))
            for kb in count:
                count[kb] /= total
            self.grammar[k]=dict(count)
    
    #create a matrix which entry detailed all the rules in order to gain time in CYK
    def build_mat_grammar(self):
        
        for k,v in self.grammar.items():
            for kb, vb in v.items():
        
                if len(kb.split("_"))==2:
                    isunary=False
                    #lhs, lrs, prob,lrs1,lrs2,isunary
                    self.mat_grammar.append([k,kb,vb,kb.split("_")[0],kb.split("_")[1],isunary])
                else:
                    isunary=True
                    self.mat_grammar.append([k,kb,vb,kb.split("_")[0],None,isunary])

     ###################Chomsky Normal Form########################
    
    #Find terminal node from the grammar in order to compute the chomsky normal form
    def find_term_node(self):
        term_node=[]
        for k,v in self.lexic.items():
            for kb in v.keys():
                term_node.append(kb)
        self.term_node=set(term_node) 
   
    
    #Eliminate non-solitary terminals
    def non_sol_term(self):
        chomsky_tree=copy.deepcopy(self.grammar)
        for k,v in self.grammar.items():
            for kb, vb in v.items():
                tmp=kb.split("_") 
                if(len(tmp)>1): 
                    for i,l in enumerate(tmp): 
                        if l in self.term_node:
                            tmp[i]="*"+l 
                            chomsky_tree["*"+l]={l:1.0} 

                    del chomsky_tree[k][kb] 
                    chomsky_tree[k]["_".join(tmp)]=vb

        self.grammar=chomsky_tree



    #Binarization
    def binarize(self):
        chomsky_tree=copy.deepcopy(self.grammar)
        for k,v in self.grammar.items():
            for kb, vb in v.items():
                tmp=kb.split("_")
                nb_split=len(tmp)-1
                if(nb_split>1): #if not we may just face an unary node problem for later
                    elt=tmp[0]
                    #split successively
                    for i in range(nb_split):

                        new_key="|".join(tmp[i+1:])
                        if i !=0:
                            chomsky_tree[old_key]={elt+"_"+new_key : 1.0} 
                        else:
                            chomsky_tree[k][elt+"_"+new_key]=vb
                        old_key=new_key
                        elt=tmp[i+1]

                    del chomsky_tree[k][kb]
        self.grammar=chomsky_tree
                    

    
    #Eliminate unary rules with non terminals POS
    def unary_rule(self):
        chomsky_tree=copy.deepcopy(self.grammar) 
     
        for k,v in self.grammar.items():
            #tous les noeuds possibles
            for kb in v.keys(): 
                if (len(kb.split("_"))==1 and kb not in self.term_node):# the rule k->kb  is unary
                        prob_trans=chomsky_tree[k][kb] # retain P: A-> B

                        del chomsky_tree[k][kb] 
                        for kt in self.grammar[kb].keys(): 

                            if chomsky_tree[k].get(kt) != None:
                                chomsky_tree[k][kt]+=prob_trans*self.grammar[kb][kt]  
                            else:
                                chomsky_tree[k][kt]=prob_trans*self.grammar[kb][kt] 
                                
        self.grammar=chomsky_tree
    
    #Erase unary non terminal rules remaining after the first unarization and renormalize (avoid cycles)
    def erase_last_unary(self):
        chomsky_tree=copy.deepcopy(self.grammar) 
        for k, v in self.grammar.items():
            normal=False
            for kb in v.keys(): 
                if len(kb.split("_"))==1 and kb not in self.term_node:
                    del chomsky_tree[k][kb]
                    normal=True
            
            #Renormalization of child of left hand rule k
            if normal:
                count=chomsky_tree[k]
                total = float(sum(count.values()))
                for kb in count:
                    count[kb] /= total
                    chomsky_tree[k]=dict(count)
       
        self.grammar=chomsky_tree
    
    

    ###################CYK Algorithm########################
   
    #Implementation of CYK algorithm with tree probabilities in order to keep the higher probability tree
    #Sentence with word in train in entry
    def PCYK(self,corrected_sentence):
        
        n_nodes=len(list(self.dic_nodes.keys()))
        n=len(corrected_sentence)
        prob = np.zeros((n,n,n_nodes))
        back = np.zeros((n,n,n_nodes),dtype=np.ndarray)


        for u,word in enumerate(corrected_sentence):
            for pos_lexic in self.lexic[word]:
                v = self.dic_nodes[pos_lexic]
                prob[0,u,v] = self.lexic[word][pos_lexic]
    
        #Rules which lead to terminals are unary
        for u in range(n):
            for line_rules in self.mat_grammar:
                    lhs, lrs, pr, lrs1,lrs2,isunary = line_rules
                    if isunary: 
                        pos_l = self.dic_nodes[lhs]
                        pos_r = self.dic_nodes[lrs]
                        prob_unary = prob[0,u,pos_r] * pr 
                        if prob[0,u,pos_r]>0 and prob_unary > prob[0,u,pos_l]: 
                            prob[0,u,pos_l] = prob_unary
                            back[0,u,pos_l] = (0,pos_r,-1)
        
        #Binary rules with 2 non terminals
        for l in progressbar.progressbar(range(2,n+1)):#length of sub sequence
            for u in range(1,n-l+2): #beginning of sub sequence
                for w in range(0,l): #where do we split
                        for line_rules in self.mat_grammar:
                            lhs, lrs, pr, lrs1,lrs2,isunary = line_rules
                            if not isunary:
                                pos_par = self.dic_nodes[lhs]
                                left = self.dic_nodes[lrs1]
                                right = self.dic_nodes[lrs2]
                                
                                prob_prod = pr
                                prob_binary = prob_prod * prob[w-1,u-1,left] * prob[l-w-1,u+w-1,right]
                               
                                if prob[w-1,u-1,left] > 0 and prob[l-w-1,u+w-1,right] > 0 and prob[l-1,u-1,pos_par] < prob_binary:
                                    
                                    prob[l-1,u-1,pos_par] = prob_binary
                                    back[l-1,u-1,pos_par] = (w,left,right)
            
        return(back, prob)
          
                            
    # Return the bracket sentence from the CYK back table   
    def build_bracket_sent(self,start_sent,back,l,start,pos,S=''): 
       
        indexes = back[l,start,pos]
        if indexes == 0: 
            return start_sent[start]
        else: 
            w,left,right = indexes
            if  left!=0 and right == -1 : 
                return "(" + list(self.dic_nodes.keys())[left]+ " " + start_sent[start] + ")"

            S += "(" +list(self.dic_nodes.keys())[left]+" "+ self.build_bracket_sent(start_sent,back,w-1,start,left) +")"
            S += "(" +list(self.dic_nodes.keys())[right]+" "+ self.build_bracket_sent(start_sent,back,l-w,start+w,right) +")" 

        return S
    
    #unchomsky a bracket sentence obtained from the build_bracket_sent function
    def unchomsky_bracket_sentence(self,res_cyk):
        while (res_cyk.find("(*")!= -1 or res_cyk.find("|")!=-1):

            if res_cyk.find("(*")== -1:
                st=res_cyk.find("|")

                for j,u in enumerate(reversed(res_cyk[:st])):
                    if u =="(":
                        break
                start=st-(j+1)

            else:
                start=res_cyk.find("(*")
            part=res_cyk[start:]
            #ok find the closing parenthesis
            score=0
            find_close_one=False
            i=-1
            while not find_close_one:
                #print(i)
                i=i+1
                if (part[i]=="("):
                    score+=1
                if (part[i]==")"):
                    score-=1
                if score==0:
                    find_close_one=True
                    end=start+i
                    res_cyk=res_cyk[:start]+" ".join(res_cyk[start:(end+1)].split()[1:])[:-1]+res_cyk[end+1:]
            
        return(res_cyk)

      
            
    



class OOV: 
    
    def __init__(self,lexic_train,path_embeddings,unigram,bigram):
        self.lexic = lexic_train
        self.bigram = bigram
        self.unigram = unigram
        words, embeddings = pickle.load(open(path_embeddings, 'rb'),encoding='latin-1')
        self.dic_embedding = {k:v for k,v in zip(words,embeddings)}
    
    #Damerau-Levenshtein distance
    def dist_DL(self,w1,w2):
        n1 = len(w1)
        n2 = len(w2)
        m=np.zeros((n1+1,n2+1))

        for i  in range(n1+1):
            m[i, 0] = i
        for j in range(n2+1):
            m[0, j] = j
        for i in range(1,(n1+1)):
            for j in range(1,(n2+1)):
                
                if w1[i-1] == w2[j-1]:
                    cost = 0
                else:
                    cost = 1

                m[i,j] = min(m[i-1,j] + 1,m[i,j-1] + 1,m[i-1,j-1] + cost)

                if( w1[i-1] == w2[j-2] and w1[i-2] == w2[j-1]): #Transposition
                    m[i,j] = min(m[i,j],m[i-2,j-2] + cost) 

        return(m[n1,n2])

    #Cosine similarity
    def dist_cosine(self,w1,w2):
        a = self.dic_embedding[w1]
        b = self.dic_embedding[w2]
        return (np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

    #Return n nearest neighbour computed with cosine similarity or DL 
    def nearest_neighbour(self,word,n,dist="dist_cosine"):
       
        if dist=="dist_cosine":
            distances = {e:self.dist_cosine(word,e) for e in self.lexic.keys() if e in self.dic_embedding.keys() }
            sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))[::-1]
        else:
            distances = {e:self.dist_DL(word,e) for e in self.lexic.keys()}
            sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))   
  
        return [s[0] for s in sorted_distances[:n]]
    
    #Return n_cosine nearest neighbour from dist cosine and n_lev nearest neighbour from DL distance
    def give_candidates_train(self,word,n_cosine,n_lev):
    
        list_cosine=[]
        if word in self.dic_embedding.keys():

            list_cosine=self.nearest_neighbour(word,n_cosine,dist="dist_cosine")

        list_levenstein=self.nearest_neighbour(word,n_lev,dist="Lev")

        return(list_cosine + list_levenstein) 
    
    #Process the whole sentence usingcandidates from the distances, unigram and bigram
    #Return a "word-in-train" sentence in order to feed PCYK algorithm
    def process_a_sentence(self,sent,n_cos,n_lev,hyper_lambda):
        sent_in_train=[]
        for i,word in enumerate(sent):
            if word in self.lexic.keys():
                sent_in_train.append(word) 
            else:  
                pmax=0
                for candidate in self.give_candidates_train(word,n_cos,n_lev):
                   
                    if i==0:
                        start="#s"
                    else:
                        start=sent_in_train[i-1] 

                    if oov.bigram[start].get(candidate)!= None:  
                        p= hyper_lambda * self.unigram[candidate]+(1-hyper_lambda)*self.bigram[start][candidate]
                    else:
                        p= hyper_lambda * self.unigram[candidate]

                    if p>pmax:
                        pmax=p
                        winner=candidate

                sent_in_train.append(winner)
       
        return(sent_in_train)






#Load corpus

filepath = parser.paths[0]

with open(filepath, 'r') as f:
    corpus = [line.strip('\n') for line in f]
        
corpus_train=corpus[:2479]
corpus_test=corpus[2789:]
corpus_dev=corpus[2479:2789]

print("corpus load")

#Grammar
print("learning corpus grammar")
train=CorpusManager(corpus_train)
train.erase_dash()
train.build_pcfg()
train.build_lexic()
train.find_term_node()

print("Put in chomsky normal form")
train.non_sol_term()
train.binarize()
train.unary_rule()
train.erase_last_unary()
train.build_mat_grammar()

train.build_nodes()
train.build_sentences()
train.build_bigram()
train.build_unigram()

#OOV
print("building OOV from corpus")
oov=OOV(lexic_train=train.lexic,path_embeddings=parser.paths[1],unigram=train.unigram,bigram=train.bigram)

print("finish")
val=CorpusManager(corpus_dev)
test=CorpusManager(corpus_test)
val.erase_dash()
test.erase_dash()
val.build_sentences()
test.build_sentences()

if parser.sentence == False:

	input_path = parser.files[0]
	

	with open(input_path,'r') as f:
	    lines = [line.strip('\n') for line in f]

	sentence_to_parse=[]
	for line in lines:
	    sentence_to_parse.append(line.split())

	
	pred_sent=[]
	print("parsing sentence in the file provided")
	for i,sent in enumerate(sentence_to_parse):
		print("sentence", i+1,"/",len(sentence_to_parse))
		start_sent=sentence_to_parse[i]
		corrected_sent=oov.process_a_sentence(start_sent,30,4,0.05)
		back, prob=train.PCYK(corrected_sent)
		sol=train.build_bracket_sent(start_sent,back,len(corrected_sent)-1,0,train.dic_nodes["SENT"]) #add start sent here
		print("CYK")
		res_cyk="( (SENT "+sol+"))"
		pred_sent.append(train.unchomsky_bracket_sentence(res_cyk))
	
	with open('parsed_sentences_from_provided_file.txt','w') as f:
		for line in pred_sent:
			f.write(line[2:][:-1])
			f.write("\n")
	
	if (parser.files[1] != "no_target"):

		gold_sent_path=parser.files[1]
		with open(gold_sent_path,'r') as f:
			gold_sent = [line.strip('\n') for line in f]

		with open('gold_path.txt','w') as f:
			for line in gold_sent:
				f.write(line[2:][:-1])
				f.write("\n")

		gold_path="gold_path.txt"
		test_path='parsed_sentences_from_provided_file.txt'
		result_path = "res_parsing_of_provided_file.txt"
		score=scorer.Scorer()
		score.evalb(gold_path, test_path, result_path)
		print("You can look at the results of parsing in the file res_parsing_of_provided_file.txt")

else:
	input_sentence=parser.sentence[0]
	print("process the sentence:",input_sentence)

	input_sentence=" ".split(input_sentence)

	corrected_sentence=oov.process_a_sentence(input_sentence,30,4,0.05)
	print("Corrected sentence with words from train:"," ".join(corrected_sentence))

	print("parsing")
	back, prob=train.PCYK(corrected_sentence)
	sol=train.build_bracket_sent(input_sentence,back,len(corrected_sentence)-1,0,train.dic_nodes["SENT"]) #add start sent here
	res_cyk="( (SENT "+sol+"))"
	print("parsing tree from the input sentence:",train.unchomsky_bracket_sentence(res_cyk))


END_SCRIPT
#ok ici j'appelle les 1 et premier arguments de ma ligne sh run.sh ag1 arg2
# Run script.py
#python script.py --corpus_path $1 --test_path $2 
#$1 $2 sequoia embeddings
python script.py -paths "$1" "$2" -sentence $3 -files $4 $5




rm script.py
