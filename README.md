## Massively Multilingual Image Dataset Tools

This repository contains scripts and tools for working with the MMID, a large dataset of images, and the words they represent, in 100 languages.

Information about the dataset can be found at the [dataset website](https://multilingual-images.org)

### Translation scripts

`src/translation/`

In the [paper](http://www.cis.upenn.edu/~ccb/publications/learning-translations-via-images.pdf) that introduced the dataset, we showed that the MMID can be used for translating words from many languages into English.
In this subdirectory, find scripts to recreate the experiments in the paper.

To replicate a translation experiment from our paper, use the following script for example, with the Latvian language, from the MMID [downloads page](http://multilingual-images.org/downloads.html):

                python code/evaluate_package_cnn_combined.py  
                    -f nlp/data/word-translation/language_packages/latvian/latvian-features/ 
                    -e nlp/data/word-translation/language_packages/latvian/english-features/ 
                    -d mmid-tools/dictionaries/
                    -o ~ 
                    -t 4
                    -tc 4

where `-t` and `-tc` specify the number of CPUs to use for loading and computing similarities.

### Dictionaries

To create the MMID, we started with crowdsourced dictionaries from the paper [The Language Demographics of Amazon Mechanical Turk](http://aclweb.org/anthology/Q14-1007).

