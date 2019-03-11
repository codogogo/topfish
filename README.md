# SemScale
A tool for semantic scaling of political text based on word embeddings.

## (Multilingual) Word Embeddings

We provide a set of pre-trained FastText embeddings for the following five language: English, French, German, Italian and Spanish, that can be obtained from here: 

https://drive.google.com/file/d/1Oy61TV0DpruUXOK9qO3IFsvL5DMvwGwD/view?usp=sharing 

Nonetheless, you can easily use the tools for texts in other languages or with different embeddings as well, as long as you:

- provide a (language-prefixed) word embedding file containing the vocabularies of new languages. Entries must be prefixed (abbreviation for the language plus double underscore "__", e.g., Bulgarian might be prefixed with "bg__")
- Update the list of supported languages in the beginning of the code file *nlp.py* and at the beginning of the task script you're using (e.g., *scaler.py*)

## Semantic Scaler (SemScale)

*python scaler.py -h*

The expected input is in the one-text-per-file format. Each text file should contain a language in the first line, i.e., the format should be "*language*\n*text of the file*". 

### Prerequisites

- All script requires the basic libraries from the Python scientific stack: *numpy* (tested with version 1.12.1), *scipy* (tested with version 0.19.0), and *nltk* (tested with version 3.2.3); 

## Other functionalities

This branch focuses only on SemScale. More information about additional functionalities (classification, topical-scaling) can be obtained by running the scripts with the *-h* option: 

1. *python trainer.py -h*
2. *python predictor.py -h*
3. *python scaler.py -h*
4. *python topical-scaler.py -h*
5. *python wordfish.py -h*

To know more, see the main branch: https://github.com/codogogo/topfish

## Referencing

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
