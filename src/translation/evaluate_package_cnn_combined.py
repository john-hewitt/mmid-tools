#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import io
import operator
import threading
import csv
import json
import argparse
from collections import defaultdict
import cPickle as pickle
import numpy as np
from scipy import sparse
from itertools import izip_longest
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from tqdm import tqdm

from find_translations import compute_MRR, compute_MRR_top_word, top_n_acc

# locks for printing to console
thread_printing_lock = threading.Lock()
multiprocessing_printing_lock = multiprocessing.Lock()

def multiprocessing_safe_print(output):
    with multiprocessing_printing_lock:
        print(output.encode('utf-8'))

def load_combined_foreign_features(filepath):
  features = pickle.load(open(filepath))
  print(features.shape)
  return features

def load_combined_english_features(filepath):
  features = pickle.load(open(filepath))
  print(features.shape)
  return features


def compute_avg_max(similarities):
    """
        Method to compute the AVG-MAX similarity score
        for a given src word img set and target img
        set. Computes similarity score for every img
        in src and averages it.
    """
    return np.mean(np.max(similarities, axis=1))

# defines the compute sim score threads so we can generate results in a concurrent way

def compute_sim_score(foreign_word, feature_set_foreign
    , target_english_words, target_word_matrix_all, target_word_indices_map):
  try:
      # also track all possibilities so we can output the top N options and their similarities
      # ordered in same way as target_word_set array
      options_for_current_word = []

      features_src = feature_set_foreign[foreign_word]

      has_src_features = features_src.shape[0] > 0 and features_src.shape[1] > 0
      if has_src_features:
          all_cosine_similarities = cosine_similarity(features_src, target_word_matrix_all)

          # Computing similarity value for all candidate translations in target lang
          for target_word in target_english_words:
              sim_val = 0.0
              if target_word in target_word_indices_map:
                  feature_indices = target_word_indices_map[target_word]
                  similarities = all_cosine_similarities[: ,feature_indices]
                  sim_val += compute_avg_max(similarities)

              options_for_current_word.append(sim_val)

          #max_index, _ = max(enumerate(options_for_current_word), key=operator.itemgetter(1))
          #max_target_word = target_english_words[max_index]

          return foreign_word, options_for_current_word
      else:
        print(foreign_word, "has no source features")
  except Exception as e:
    print("Ack!", foreign_word)
    print e

def get_english_path_suffixes(dictionary_base_dir, possible_english_words):
    dict_paths = [dictionary_base_dir+"english.superset"+"{0:0=2d}".format(i+1) for i in range(0, 27)]
    word_map = {}
    for i, dict_path in enumerate(dict_paths):
        # Open the dictionary file
        with io.open(dict_path, 'r', encoding='utf-8') as dict_f:
            # Form an in-memory map of true src-target
            # language mappings using the dictionary
            for j, line in enumerate(dict_f.readlines()):
                english_word = line.strip()
                if english_word in possible_english_words:
                    word_map[english_word] = "English-" + "{0:0=2d}".format(i+1) + "/" + str(j)
    return word_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate CNN image features')
    parser.add_argument('-f', '--foreign_language_features', help='src/package dir for foreign word features')
    parser.add_argument('-e', '--english_language_features', help='base dir for English language features')
    parser.add_argument('-td', '--temp_data_dir', help='destination dir for temp data')
    parser.add_argument('-d', '--dictionaries_dir', help='dir that contains the bilingual dictionaries')
    parser.add_argument('-l', '--word_limit', type=int, help='word limit for testing purposes')
    parser.add_argument('-i', '--image_limit', type=int, help='word limit for testing purposes')
    parser.add_argument('-o', '--output_dir', help='output dir with result files')
    parser.add_argument('-t', '--num_threads', default=64, type=int, help='number of threads to use')
    parser.add_argument('-tc', '--num_threads_cosine_sim', default=64, type=int, help='number of threads to use')
    parser.add_argument('-mf', '--metadata_filepath',  default = '',
        help='google image crawl JSON metadata file (needed for language-confident eval)')
    parser.add_argument('-la', '--language_analysis_filepath', default = '',
        help='TSV of google:ru URL to is-in-language boolean (needed for language-confident eval)')
    opts = parser.parse_args()

    threadLimiter = threading.BoundedSemaphore(opts.num_threads)
    threadLimiterCosineSim = threading.BoundedSemaphore(opts.num_threads_cosine_sim)
    multiprocessingPool = multiprocessing.Pool(opts.num_threads)
    threadingPool = multiprocessing.pool.ThreadPool(opts.num_threads)

    foreign_language_feature_dir = opts.foreign_language_features
    if not foreign_language_feature_dir.endswith('/'):
        foreign_language_feature_dir += '/'

    target_language_feature_dir = opts.english_language_features
    if not target_language_feature_dir.endswith('/'):
        target_language_feature_dir += '/'

    foreign_language_name = foreign_language_feature_dir.split('/')[-2]

    output_dir = opts.output_dir
    if not output_dir.endswith('/'):
        output_dir += '/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # directory of dictionaries in multilingual-google-image-scraper repo
    dictionaries_dir = opts.dictionaries_dir
    if not dictionaries_dir.endswith('/'):
        dictionaries_dir += '/'

    #with io.open(os.path.dirname(os.path.realpath(__file__)) + "/language-code-map.json", 'r', encoding='utf-8') as f:
    with io.open(os.path.join(dictionaries_dir, 'language-code-map.json'), 'r', encoding='utf-8') as f:
        language_code_map = json.load(f)
    lower = {x.lower():y for x,y in language_code_map.items()}
    language_code_map.update(lower)
    dictionary_extension = language_code_map[foreign_language_name.split('-')[0].lower()]
    foreign_dictionary_file = dictionaries_dir + "dict." + dictionary_extension

    new_foreign_words = set()
    target_english_words = set()
    foreign_word_english_map = defaultdict(list)
    english_word_foreign_word_reverse_map = defaultdict(list)
    foreign_word_index_word_map = {}
    with io.open(foreign_dictionary_file, 'r', encoding='utf8') as new_dict_f:
        for i, line in enumerate(new_dict_f.readlines()):
            cells = line.split("\t")
            foreign_word = cells[0].strip()
            new_foreign_words.add(foreign_word)
            english_words = cells[1:]
            foreign_word_index_word_map[str(i)] = foreign_word
            for english_word in english_words:
                english_word = english_word.strip()
                target_english_words.add(english_word)
                english_word_foreign_word_reverse_map[english_word].append(foreign_word)
                foreign_word_english_map[foreign_word].append(english_word)

            if opts.word_limit and i+1 >= opts.word_limit:
                break

    print("Loaded " + str(len(foreign_word_english_map.keys())) + " words for the foreign dictionary")

    english_path_suffix_map = get_english_path_suffixes(dictionaries_dir, target_english_words)

    feature_set_english = {}
    feature_set_foreign = {}

    url_language_map = {}
    if opts.metadata_filepath:
      try:
        with open(opts.language_analysis_filepath ) as fin:
          for line in fin:
            url, inlang_str = [x.strip().decode('utf-8') for x in line.split('\t')]
            url_language_map[url] = True if inlang_str == '1' else False
          print "Loaded %d URLs into language confidence dictionary"%len(url_language_map)
      except IOError:
        print "Language-confidence map load failed. Not being used."

    
    ##### Prepping the file paths for foreign feature file loading

    print "Loading foreign combined features"
    for i, foreign_word in tqdm(foreign_word_index_word_map.items()):
      path_to_features_for_word = foreign_language_feature_dir + i + '.pkl'
      feature_set_foreign[foreign_word] = load_combined_foreign_features(path_to_features_for_word)

    print "Loading english combined features"
    for english_word, path_suffix in tqdm(english_path_suffix_map.items()):
        print(path_suffix,english_word)
        path_to_features_for_word = target_language_feature_dir + path_suffix + '.pkl'
        feature_set_english[english_word] = load_combined_english_features(path_to_features_for_word)

    all_similarity_scores = {}

    target_word_indices_map = {}
    target_word_feature_vectors = []
    current_feature_index = 0
    for target_word in target_english_words:
        if target_word in feature_set_english:
            word_features = feature_set_english[target_word]
            has_target_features = word_features.shape[0] > 0 and word_features.shape[1] > 0

            if has_target_features:
                next_feature_index = current_feature_index + word_features.shape[0]
                target_word_feature_vectors.append(word_features)
                target_word_indices_map[target_word] = range(current_feature_index, next_feature_index)
                current_feature_index = next_feature_index

    print("Starting vstack operation for " + str(len(target_word_feature_vectors)) + " sparse matrices")

    target_word_matrix_all = sparse.vstack(target_word_feature_vectors)

    print("Finished vstack operation for " + str(len(target_word_feature_vectors)) + " sparse matrices")

    english_word_list = list(target_english_words)

    target_word_index_mapping = {}
    for i, target_word in enumerate(english_word_list):
        target_word_index_mapping[target_word] = i

    ### New Sim Computation

    def group_sim_calls(generator,k):
      buf = []
      for i in generator:
        buf.append(i)
        if len(buf) == k:
          yield buf
          buf = []
      if buf:
        yield buf

    def generate_sim_calls():
      for i, foreign_word in foreign_word_index_word_map.items():
        yield (foreign_word, feature_set_foreign, english_word_list, target_word_matrix_all, target_word_indices_map)

    for group in tqdm(group_sim_calls(generate_sim_calls(),100)):
      for result in [threadingPool.apply_async(compute_sim_score, x) for x in group]:
        retval = result.get()
        if retval:
          (foreign_word, scores) = retval
          all_similarity_scores[foreign_word] = scores

    print "Done computing stuff! MRR time."
    ranks, mrr_val = compute_MRR(feature_set_foreign, foreign_word_english_map, all_similarity_scores, target_word_index_mapping)
    print ranks
    print mrr_val

    output_dict = {
        'ranks': ranks,
        'mrr': mrr_val
    }

    accuracy_tests = [1, 5, 10, 20, 100]
    for top_n_val in accuracy_tests:

        top_n_accuracy = top_n_acc(top_n_val, ranks)
        print("Top " + str(top_n_val) + " results score is " + str(top_n_accuracy))
        output_dict['top_'+str(top_n_val)+'_accuracy'] = top_n_accuracy

    with io.open(output_dir + "target_word_set.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(english_word_list, ensure_ascii=False)))

    with io.open(output_dir + "results.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(output_dict, ensure_ascii=False)))

    with io.open(output_dir + "all_sim_scores.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(all_similarity_scores, ensure_ascii=False)))
