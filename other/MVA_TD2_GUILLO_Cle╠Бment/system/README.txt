#In order to run the run.sh file you have 3 possibility.

1) If you want to see one sentence of your choice parsed, the command is:

sh run.sh "treebank_path" "embedding_path" "True" "sentence"

for example : sh run.sh "sequoia.txt" "polyglot-fr.pkl" "True" "Le chat mange la souris."



2) If you want to parse sentences in a text file (one sentence per line and a space between each token) the command is:

a) If you just want to obtain the parsed sentences from your file:

sh run.sh "treebank_path" "embedding_path" "False" "sentences_file_path" "no_target"

ex: sh run.sh "sequoia.txt" "polyglot-fr.pkl" "False" "sentences_to_parse.txt" "no_target"

The output will be on a file called "parsed_sentences_from_provided_file.txt" in the folder system

b) If you just want to obtain the parsed sentences from your file and compare the obtained trees with trees you already have:

ex: sh run.sh "sequoia.txt" "polyglot-fr.pkl" "False" "sentences_to_parse.txt" "target_trees.txt"

The output will still be on a file called "parsed_sentences_from_provided_file.txt" in the folder system and an additional file of results "res_parsing_of_provided_file.txt" computed with evalb will also appear.


Some remarks:

#the tree bank file is the sequoia file provided fort the TP in a txt format.
#The True/False parameter is a string so you have to put "True" or "False" for the execution, it indicates the sentence mode ("True") or the file mode ("False").

#if the text files you want to use are in the system folder it will works but if it is in another location you have to add the location before the name of the file name -> "User/../sequoia.txt"

#the target file used to compute the score between the predicted POS tags and the target POS tags must contains parsed sentence in the same format than the parsed sentences on the treebank : 

( (SENT (PONCT -)(NP (NC Février)(NC 2005))(PONCT :)(NP (DET le)(NC parquet)(PP (P de)(NP (NPP Paris))))(VN (V requiert))(NP (DET un)(PREF non-)(NC lieu)(PP (P en_faveur_de)(NP (NPP Jean)(NPP Tiberi)))(PONCT ,)(VPpart (VPP accordé)(PP (P par)(NP (DET le)(NC juge)(NPP Armand)(NPP Riberolles)))))(PONCT .)))

Then during the comparison process, a temporary file "gold_path.txt" will be created and used for evalb.


#Finally, at the end of your first execution a file "script.py" file containing my code will be created in the system folder. It can be removed with writting "rm script.py" at the end of the run.sh file.


