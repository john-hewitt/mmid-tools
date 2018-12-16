'''
Creates a foreign-language word index and translation index
based on the Pavlick et al. dictionaries and the English
index.
'''

import argprase
import glob
import os

argp = argparse.ArgumentParser()
argp.add_argument('dictionary_dir')
argp.add_argument('output_dir')
args = argp.parse_args()

langpath = os.path.join(args.dictionary_dir, 'lang2name.txt')
id2lang = {x.split('\t')[0]:x.strip().split('\t')[1] for x in open(langpath)}
eng2uuid = {x.split('\t')[0]:x.strip().split('\t')[-1] for x in open('english-vocab.tsv')}
for dictionary_path in glob.glob(args.dictionary_dir + '/dict.*'):
    for index, line in enumerate(dictionary_path):
        fields = line.strip().split('\t')
