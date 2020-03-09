import io
import nltk

def read(corpus_path):
    """Extract data from the constituency trees.
    Ignore all functional labels.
    
    Parameters
    ----------
    path : str
        path to data

    Returns
    -------
    data: list 
        list of consituency trees
    """
    
    data = []
    with io.open(corpus_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split(" ")
            line = [word.split("-")[0] if word[0]=='(' else word for word in line]
            data.append(" ".join(line))
    return data



def split(corpus, train=0.8, val=0.1, test=0.1, random_state=None):
    """Split corpus into (optional random) train, validation and test subsets

    Parameters
    ----------
    curpus : list
        list of sentences
        
    train_size : float
        Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
        Must have train_size + val_size + test_size = 1

    val_size : float
        Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
        Must have train_size + val_size + test_size = 1
        
    test_size : float
        Should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
        Must have train_size + val_size + test_size = 1
        
    random_state : int or None (default=None)
        If None, corpus is not shuffled
        If int, set random seed to random_state

    Returns
    -------
    [train, val, test] : list
        List containing train-val-test split of corpus.
    """

    data = corpus.copy()
    n = len(data)

    if train + val + test != 1:
        raise ValueError("Fractions does not add to 1.")
        
    if random_state is not None:
        shuffle(data)
        
    trainX = data[:int(n*train)+1]
    valX = data[int(n*train)+1:int(n*(train+val))+1]
    testX = data[int(n*(train+val))+1:]
    
    return [trainX, valX, testX]




def build_vocabulary(X):
    """Extract individual words from a list of list of strings

    Parameters
    ----------
    curpus : list
        list of list of words
        

    Returns
    -------
    vocabulary : set
    """

    flatten = lambda l: [item for sublist in l for item in sublist]
    return set(flatten(X))


import nltk


def extract_sentences(X):
    """Extract sentences as a list of string.
    In a constituent tree, words are the leaves.
    Use NLTK to build the tree and extract the leaves.
    
    Parameters
    ----------
    X : list
        list of sentences in bracketed formats

    Returns
    -------
    sentences : list
        List of list of words for each sentence from X
    """
       
    sentences = list()
    for s in X:
        # build contituency tree
        t = nltk.tree.Tree.fromstring(s)
        # extract leaves
        sentences.append(t.leaves())
    return sentences







