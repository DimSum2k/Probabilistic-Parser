import argparse

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_data', help='path to data', default="sequoia-corpus+fct.mrg_strict")
	parser.add_argument('--path_embeddings', help='path to embeddings', default="polyglot-fr.pkl")
	parser.add_argument('--use_multiprocessing', help='if True use multi processing', default=True)
	parser.add_argument('--n_cpus', help='number of cpus for multiprocessing', default=-1)

	parser.add_argument('--train_frac', default=0.9, help="percentage for the train set when splitting the data")
	parser.add_argument('--val_frac', default=0.0, help="percentage for the validation set when splitting the data")
	parser.add_argument('--test_frac', default=0.1, help="percentage for the test set when splitting the data")

	parser.add_argument('--n1', default=2, help="number of neighbours for k-nearest levenshtein neighbours")
	parser.add_argument('--n2', default=20, help="number of neighbours for k-nearest cosine simalarity embedding neighbours")
	parser.add_argument('--l', default=0.2, help="linear interpolation smoothing coefficient")
	parser.add_argument('--damerau', default=True, help="If true use damerau-levenshtein distance")

	return parser