import nltk
import numpy as np

def parsePCYK(init, new, pcfg):
	P,back  = P_CYK(new, pcfg) 
	S = un_chomsky(build_tree(back,init,pcfg))
	return S

def P_CYK(sentence,pcfg):
    
    r = len(pcfg.non_terminals) 
    n = len(sentence)
    P = np.zeros((n,n,r))
    back = np.empty((n,n,r),dtype=np.ndarray)

    for s,word in enumerate(sentence):
        for pos in pcfg.lexicon_[word]:
            v = pcfg.pos2index[pos]
            P[0,s,v] = pcfg.lexicon_[word][pos]
    
    for s in range(n):
        for prod in pcfg.pcfg_._productions:
            if len(prod._rhs)==1:
                v1 = pcfg.pos2index[prod._lhs]
                v2 = pcfg.pos2index[prod._rhs[0]]
                prob_transition = P[0,s,v2] * prod._ProbabilisticMixIn__prob
                if P[0,s,v2]>0 and prob_transition > P[0,s,v1]: 
                    P[0,s,v1] = prob_transition
                    back[0,s,v1] = (0,v2,None)
                    
    for l in range(2,n+1): 
        for s in range(1,n-l+2): 
            for p in range(0,l):
                for prod in pcfg.pcfg_._productions:
                    if len(prod._rhs)==2: 
                        a = pcfg.pos2index[prod._lhs]
                        Rb = prod._rhs[0]
                        Rc = prod._rhs[1]
                        b = pcfg.pos2index[Rb]
                        c = pcfg.pos2index[Rc]
                        prob_prod = prod._ProbabilisticMixIn__prob
                        
                        prob_splitting = prob_prod * P[p-1,s-1,b] * P[l-p-1,s+p-1,c]
                        if P[p-1,s-1,b] > 0 and P[l-p-1,s+p-1,c] > 0 and P[l-1,s-1,a] < prob_splitting:
                            #print(prod)
                            P[l-1,s-1,a] = prob_splitting
                            back[l-1,s-1,a] = (p,b,c)
    
    return P,back 


def build_tree(backp,sentence,pcfg,length=-1,start=0,pos=0): 
    S = ''
    indexes = backp[length,start,pos]
    if indexes is None: 
        return sentence[start]
    else: 
        p,b,c = indexes
        if c is None and not b is None: 
            return "(" + pcfg.non_terminals[b]._symbol+ " " + sentence[start] + ")"
        S += "(" +pcfg.non_terminals[b]._symbol+" "+ build_tree(backp,sentence,pcfg,p-1,start,b) +")"
        S += "(" +pcfg.non_terminals[c]._symbol+" "+ build_tree(backp,sentence,pcfg,length-p,start+p,c) +")" 
            
    return S


def correct_string(string):
    return '( (SENT' + string + '))'


def un_chomsky(bracket_string):
    
    tree = nltk.tree.Tree.fromstring(correct_string(bracket_string))
    nltk.treetransforms.un_chomsky_normal_form(tree)
    result = tree.pformat().replace('\n','')

    return result