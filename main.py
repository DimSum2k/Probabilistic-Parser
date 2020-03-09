
from pprint import pprint, pformat

from utils import read, split, build_vocabulary
from TreeManip import extract_sentences
from PCFG import PCFG
from OOV import Word2Vec, get_POS
from config import get_arguments


if __name__=="__main__":
	
	parser = get_arguments()
	opt = parser.parse_args()

	corpus = read(opt.path_data)
	print(len(corpus))
	print(corpus[3])






