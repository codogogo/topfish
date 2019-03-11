# SemScale
An easy-to-use tool for semantic scaling of political text based on word embeddings. 

## How to use it

Clone the project, then cd into the SemScale directory. The script scaler.py needs just the following inputs:

 datadir -> A path to the directory containing the input text
                        files for scaling (one score will be assigned per
                        file).
                        
  embs -> A path to the file containing pre-trained word
                        embeddings
                        
  output -> A file path to which to store the scaling results.

optional arguments:

  -h, --help -> show this help message and exit
  
  --stopwords STOPWORDS -> A file to the path containing stopwords

### Input Files

The expected input is in the one-text-per-file format. Each text file should contain a language (e.g., "en") in the first line, i.e., the format should be "*language*\n*text of the file*". 

### (Multilingual) Word Embeddings

For an easy set-up, we provide pre-trained FastText embeddings in a single file for the following five language: English, French, German, Italian and Spanish, that can be obtained from here: 

https://drive.google.com/file/d/1Oy61TV0DpruUXOK9qO3IFsvL5DMvwGwD/view?usp=sharing 

Nonetheless, you can easily use the tool for texts in other languages or with different word embeddings, as long as you:

1) provide a (language-prefixed) word embedding file, the following way: for each word, abbreviation for the language plus double underscore plus word and then the word embedding. For instance, each word in a Bulgarian word embeddings file might be prefixed with "bg__")

2) in case you employ embeddings in a different language to the 5 listed above, update the list of supported languages in the beginning of the code file *nlp.py* and at the beginning of the task script you're using (e.g., *scaler.py*)

### Output File

A simple .txt, which will be filled with filename, positional-score for each input file.

### (Optional) Stopwords

Stopwords can be automatically excluded, via this input file (one stop-word per line).

### Prerequisites

- All script requires the basic libraries from the Python scientific stack: *numpy* (tested with version 1.12.1), *scipy* (tested with version 0.19.0), and *nltk* (tested with version 3.2.3); 

### Run it!

In the SemScale folder, just run the following command:

python scaler.py path-to-embeddings-file path-to-input-file-folder output.txt

### Other functionalities

We also present a Python implementation of the famous Wordfish algorithm for text scaling. To know more, just run: 

python wordfish.py -h

Additional functionalities (classification, topical-scaling) are available in the main branch of this project: https://github.com/codogogo/topfish

### Referencing

If you're using this tool, please cite the following paper: 

```
@InProceedings{glavavs-nanni-ponzetto:2017:EACLshort,
  author    = {Glava\v{s}, Goran  and  Nanni, Federico  and  Ponzetto, Simone Paolo},
  title     = {Unsupervised Cross-Lingual Scaling of Political Texts},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {688--693},
  url       = {http://www.aclweb.org/anthology/E17-2109}
}
```
