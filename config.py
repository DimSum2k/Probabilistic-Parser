import argparse

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_data', help='path to data', default="sequoia-corpus+fct.mrg_strict")
	parser.add_argument('--path_embeddings', help='path to embeddings', default="polyglot-fr.pkl")
	parser.add_argument('--train_frac', default=0.8, help="percentage for the train set when splitting the data")
	parser.add_argument('--val_frac', default=0.1, help="percentage for the validation set when splitting the data")
	parser.add_argument('--test_frac', default=0.1, help="percentage for the test set when splitting the data")

	return parser