'''
Takes one of Brendan Callahan's packaged image sets
produces a summarized version of the metadata of each word
in jsonl format, also removing extraneous info
'''

import argparse
import os
import sys
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('input_directory')
parser.add_argument('output_filepath')
args = parser.parse_args()

with open(args.output_filepath, 'w') as fout:
    for filename in tqdm(os.listdir(args.input_directory)):
        output_dict = {}
        dir_path = os.path.join(args.input_directory, filename)
        if not os.path.isdir(dir_path):
            continue
        metadata_path = os.path.join(dir_path, 'metadata.json')
        word_path = os.path.join(dir_path, 'word.txt')
        metadata = json.load(open(metadata_path))
        word = open(word_path).read().strip()
        image_dict = {}
        text_dict = {}
        thumbnail_dict = {}
        for key in metadata:
            image_dict[key] = metadata[key]['image_link']
            thumbnail_dict[key] = metadata[key]['google']['tu']
            text_dict[key] = metadata[key]['google']['ru']

        output_dict['word_string'] = word
        output_dict['word_index'] = filename
        output_dict['image_original_urls'] = image_dict
        output_dict['image_thumbnail_urls'] = thumbnail_dict
        output_dict['webpage_urls'] = text_dict
        fout.write(json.dumps(output_dict))
        fout.write('\n')
