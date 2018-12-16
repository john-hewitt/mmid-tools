'''
Takes English package folders and constructs an index of the form

word_string\tpackage_index\tintra_package_id\tuniversal_id
'''

import os
from tqdm import tqdm
import sys

root = sys.argv[1]

english_folders = ['English-{0:02d}'.format(x) for x in range(1,28)]


with open('english-vocab.tsv', 'w') as fout:
    fout.write('\t'.join(['word_string', 'package_index', 'intra_package_id', 'uuid'])+'\n')
    for package_index in tqdm(range(1,28), desc='[eng package]'):
        english_package_name = 'English-{0:02d}'.format(package_index)
        for word_index in tqdm(range(0,10000), desc='[word index]'):
            path = os.path.join(root, english_package_name, str(word_index))
            if not os.path.exists(path):
                print('No word at {}'.format(english_package_name, str(word_index)))
                continue
            word_string = open(os.path.join(path, 'word.txt')).read().strip()
            uuid = (package_index-1)*10000 + word_index
            fout.write('\t'.join([word_string, str(package_index), str(word_index), str(uuid)])+'\n')


