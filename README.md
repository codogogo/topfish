# TopFish
A set of tools for monolingual and cross-lingual topical text classification and scaling

TopFish is a tool for topical classification and scaling of texts, primarily designed for researcher in social sciences (especially political science). It allows researchers to: 

1. Train their own topical classifier (based on a convolutional neural network) from their annotated data. (script *trainer.py*) 
2. Predict the topics for new texts using a previously trained topical classifier (script *predictor.py*)
3. Scale the collection of texts, i.e., position texts by assigning them a score in a single-dimensional space (script *scaler.py*)
4. Topically scale texts -- by first topically classifying the texts and then scaling independently texts for each topic, i.e., by taking into consideration for classification only the parts of the texts belonging to the specific topic (script *topical-scaler.py*)

Each of the above four functionalities has a corresponding command-line tool. In addition, we make available our Python re-implementation of WordFish (script *wordfish.py*), a widely used scaling algorithm based on symbolic (i.e., term-based) representations of text. In contrast, our own scaling algorithm (scripts *scaler.py* and *topical-scaler.py*) make use of semantic representations of text (word embeddings), and are, applicable in both monolingual and multilingual/cross-lingual settings.

## (Multilingual) Word Embeddings

All tools except our reimplementation of WordFish (i.e., *trainer.py*, *predictor.py*, *scaler.py*, and *topical-scaler.py*) require a file with pre-trained word embeddings as input. An embedding of a word is a numeric vector that capture the meaning of the word. The entries in the pre-trained word embeddings file need to be language prefixed. 

By default, we support five big languages: English (prefix "en__"), German (prefix "de__"), Italian (prefix "it__"), French (prefix "fr__"), and Spanish (prefix "es__"). We provide a set of pre-trained FastText embeddings for these five languages, merged into the same multilingual embedding space, that can be obtained from here: 

https://drive.google.com/file/d/1Oy61TV0DpruUXOK9qO3IFsvL5DMvwGwD/view?usp=sharing 

Nonetheless, you can easily use the tools for texts in other languages well, as long as you:

- provide a (language-prefixed) word embedding file containing the vocabularies of new languages. Entries must be prefixed (abbreviation for the language plus double underscore "__", e.g., Bulgarian might be prefixed with "bg__")
- Update the list of supported languages in the beginning of the code file *nlp.py* and at the beginning of the task script you're using (e.g., *scaler.py*)

## Task Scripts

There is a separate command-line script for each of the four tasks. The descriptions of mandatory and optional arguments of these scripts can be obtained by running the scripts with the *-h* option: 

1. *python trainer.py -h*
2. *python predictor.py -h*
3. *python scaler.py -h*
4. *python topical-scaler.py -h*
5. *python wordfish.py -h*

### Prerequisites

- All script requires the basic libraries from the Python scientific stack: *numpy* (tested with version 1.12.1), *scipy* (tested with version 0.19.0), and *nltk* (tested with version 3.2.3); 
- Running scripts *trainer.py*, *predictor.py*, and *topical-scaler.py* require having the TensorFlow library (tested with version 1.3.0) installed. 

## Input Data Formats

### Topical classification

The default format for input texts for classification and predicting topics (scripts *trainer.py* and *predictor.py*) is the **short text format**. The short text format assumes that each line of each input file contains one text instance, in the following format: *language*\t*topic-label*\t*text of the line*. Alternatively, by setting the *-ltf* flag to *True* (scripts *trainer.py* and *predictor.py*), you can indicate that your input directory contains the training/testing instances in the **long text format**. This format assumes that each file is a single text instance (as opposed to each line of each file in the short text format) and the structure of the file must be *language*\t*topic-label*\t*text of the whole file*. 

In both formats, the *topic-label* must be a single token (that is, one string with any whitespace characters) and the *language* for each text instance must be one of the supported languages (by default, using the abovementioned multilingual embeddings, we support: 'en', 'de', 'fr', 'es', and 'it'). In case that you don't have the gold standard topic labels for the texts you're using  

### Scaling

For scaling tasks (*scaler.py*, *topical-scaler.py*) the expected input is in the one-text-per-file format. Each text file should contain a language in the first line, i.e., the format should be "*language*\n*text of the file*". 

For the wordfish scaler (script *wordfish.py*) the input are only the raw text files. As the WordFish scaling model is symbolic, it cannot work in the multilingual setting and there is no need to specify the language of the text in each file. Using the script *wordfish.py* it only makes sense to scale texts that are all in the same language. 

## Referencing

If you're using the scaling functionality (either TopFish scaling or our reimplementation of the WordFish scaler), please cite the following paper: 

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

If you're using the classification functionality in the work you publish, please cite the following paper: 

```
@InProceedings{glavavs-nanni-ponzetto:2017:NLPandCSS,
  author    = {Glava\v{s}, Goran  and  Nanni, Federico  and  Ponzetto, Simone Paolo},
  title     = {Cross-Lingual Classification of Topics in Political Texts},
  booktitle = {Proceedings of the Second Workshop on NLP and Computational Social Science},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {42--46},
  url       = {http://www.aclweb.org/anthology/W17-2906}
}
```




