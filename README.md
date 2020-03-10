## Probabilistic Parser for French

This system provides a Probabilistic Parser for French based on the CYK algorithm, and the PCFG model and that is robust to unknown words thanks an OOV (out-of-vocabulary) module.

### System arguments
Input corpus parameters :
* `--path_data`: path to data
    * By default : "data/sequoia-corpus+fct.mrg_strict" (SEQUOIA treebank)
* `--train_frac`: percentage for the train set when splitting the data
    * By default : 1.0 (100%)
* `--val_frac`: percentage for the validation set when splitting the data
    * By default : 0 (0%)
* `--test_frac`: percentage for the test set when splitting the data
    * By default : 0.0 (0%)

OOV parameters :
* `--path_embeddings`: path to embeddings
    * By default : 'data/polyglot-fr.pkl' (Polyglot French embeddings)
* `--n1`: number of neighbours for k-nearest levenshtein neighbours
    * By default : 2
* `--n2`: number of neighbours for k-nearest cosine simalarity embedding neighbours
    * By default : 20
* `--l`: linear interpolation smoothing coefficient in language model
    * By default : 0.2
* `--damereau`: If true use damerau-levenshtein distance
    * By default : True
    
CYK parameters :
* `--write_to_parse`: directly parse sentence written in terminal
    * By default : None
* `--file_to_parse`: Path to the file containing sentences to parse (one sentence per line, exactly one whitespace between each token)
    * By default : "data/test.txt"
* `--output_path`: Output path for the sentences parses outputs.
    * By default : "results/parse_results.txt"
    
Multiprocessing parameters:
* `--use_multiprocessing`: 'call to use multi processing'
* `--n_cpus`: number of cpus for multiprocessing. If -1, use all the CPUs available.
    * By default : -1

### How to use
Here are some examples to use the system proposed :

1. To train the parser on 100% of the SEQUOIA treebank and parse "data/test.txt" **with** multiprocessing and write the results to the file "results/parse_results.txt":
```
sh run.sh --use_multiprocessing
```

2. To train the parser on 100% of the SEQUOIA treebank and parse the last 10% **without** multiprocessing and and write the results to the file "results/parse_results.txt":
```
sh run.sh 
```

3. To train the parser on 100% of the SEQUOIA treebank and parse a written sentence and write the results to the file "results/parse_results.txt":
```
sh run.sh --write_to_parse "Aucun financement politique occulte n' a pu être mis en évidence ."
```

4. To train the parser on 100% of the SEQUOIA treebank and parse the sentences in a file and write the results to the file "results/parse_results.txt":
```
sh run.sh --file_to_parse "data/test.txt" --output_path "results/parse_results.txt"
```
