#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import argparse
from collections import defaultdict
import operator
import threading
import multiprocessing
import json
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import cProfile

# lock for printing to console
printing_lock = threading.Lock()

word_skip_files = set(["word.txt", "errors.json", "metadata.json"])

# for 4096 hex triples, this is the full set of possible keys
hex_string_array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
histogram_hex_keys = [a+b+c for a in hex_string_array for b in hex_string_array for c in hex_string_array]
histogram_hex_array_index_map = {k: i for i, k in enumerate(histogram_hex_keys)}

DEBUG = False

def top_n_acc(N, ranks):
    """
        Calculates the proportion of the src words
        that had correct translatiions in top-N
        ranked translations

    :param N: Top N value to be considered
    :param src_word_set: Set of src language words [for compute_MRR]
    :param target_word_set: Set of target language words [for compute_MRR]
    :param src_feat_map: Feature classes from src lang images [for compute_MRR]
    :param target_feat_map: Feature classes from target lang images [for compute_MRR]
    :param lambdaa: Weights for each feature class [for compute_MRR]
    :param word_map: dictionary of target lang [for compute_MRR]
    :param ranks: Pre-computed ranks of candidate translations

    :return: top-n accuracy score
    """
    # Calculate the proportion of words with
    # correct translation in the top N ranks
    top_n_cnt = 0
    for w in ranks:
        if ranks[w] <= N:
            top_n_cnt += 1

    return (top_n_cnt * 1.0) / len(ranks.keys())


# defines the compute top mrr threads so we can calculate mrr scores in a concurrent way
def compute_top_word_mrr_rank(src_word, src_word_similarity_scores, target_word_index_mapping, src_word_translations, english_word_foreign_word_reverse_map):
    target_word_sim_mapping = {w: src_word_similarity_scores[i] for w, i in target_word_index_mapping.items()}
    sorted_target_word_sim_tuples = sorted(target_word_sim_mapping.items(), key=lambda x: x[1], reverse=True)
    target_word_rank_mapping = {w[0]: i + 1 for i, w in enumerate(sorted_target_word_sim_tuples)}

    # find the best ranked translation for each of the possible translations for this word
    best_rank = 1000000
    for possible_translation in src_word_translations:
        if possible_translation not in target_word_rank_mapping:
            continue
        current_translation_rank = target_word_rank_mapping[possible_translation]
        if current_translation_rank < best_rank:
            best_rank = current_translation_rank

    # go over all of the translations that appear before this best-ranked word, and if there are multiple translation
    # options for other words that appear before the best translation for the current word, adjust the rank downwards
    user_src_word_mapping = defaultdict(list)
    initial_best_rank = best_rank
    for i, word_sim_tuple in enumerate(sorted_target_word_sim_tuples):
        # break the for-loop iteration when we hit the initial best rank spot in the list
        if i >= initial_best_rank - 1:
            break

        english_word, _ = word_sim_tuple

        # for this english word, go through all of the possible foreign words it is a translation for,
        # and for each foreign word that has had more than one translation appear, decrement the best rank by 1
        possible_foreign_words = english_word_foreign_word_reverse_map[english_word]
        num_to_decrement = 0
        for possible_foreign_word in possible_foreign_words:
            user_src_word_mapping[possible_foreign_word].append(english_word)
            if len(user_src_word_mapping[possible_foreign_word]) > 1:
                num_to_decrement += 1
        best_rank -= num_to_decrement

    # try and flush out these scenarios
    if best_rank < 1:
        print(("WARNING: best rank was less than 1 for " + src_word).encode('utf-8'))

    # n.b. guard against rare issue where rank gets decremented too far, overall this should be a wash
    best_rank = max(best_rank, 1)

    return src_word, best_rank

def compute_MRR_top_word(src_word_set, word_map, all_similarity_scores, target_word_index_mapping, english_word_foreign_word_reverse_map, multiprocessingPool):
    """
    Computes the MRR measure to evaluate the performance
    of bilingual induction using src and target word
    img sets using the specified dictionary
    """

    # Traverse through all source words and find the
    # best match among translations
    ranks = {}
    total_ranked_words = 0
    # pool_tasks = []

    for src_word in src_word_set:
        # Check for presence of src word in the
        # word map with src-target word mappings
        if src_word not in word_map or src_word not in all_similarity_scores:
            print(
            ("Dictionary doesn't contain " + unicode(src_word) + " . Can't compute MRR for the word\n").encode('utf-8'))
            continue
        #
        # process_params = (src_word, all_similarity_scores[src_word], target_word_index_mapping,
        #                                          word_map[src_word], english_word_foreign_word_reverse_map)
        # pool_tasks.append(process_params)

        foreign_word, word_rank = compute_top_word_mrr_rank(src_word, all_similarity_scores[src_word],
                                                            target_word_index_mapping, word_map[src_word],
                                                            english_word_foreign_word_reverse_map)
        ranks[foreign_word] = word_rank

        #print(("src word - " + src_word + " target word - " + unicode(word_map[src_word]) + " has rank " + unicode(best_rank)).encode('utf-8'))
        total_ranked_words += 1

    # print("Starting " + str(len(pool_tasks)) + " processes to load features for the foreign words")
    #
    # pool_results = [multiprocessingPool.apply_async(compute_top_word_mrr_rank, t) for t in pool_tasks]
    # for result in pool_results:
    #     (foreign_word, word_rank) = result.get()
    #     ranks[foreign_word] = word_rank
    #
    # print("Finished " + str(len(pool_tasks)) + " processes to load features for the foreign words")

    # Calculate Mean reciprocal rank
    mrr_val_numerator = 0.0
    for rank in ranks.values():
        mrr_val_numerator += 1.0 / rank
    mrr_val = mrr_val_numerator / total_ranked_words

    return ranks, mrr_val


def compute_MRR(src_word_set, word_map, all_similarity_scores, target_word_index_mapping):
    """
        Computes the MRR measure to evaluate the performance
        of bilingual induction using src and target word
        img sets using the specified dictionary

    """
    all_ranks = {}
    total_ranked_translation_pairs = 0
    failed_english_words = set()
    mrr_val_numerator = 0.0
    for src_word in src_word_set:
        # Check for presence of src word in the
        # word map with src-target word mappings
        if src_word not in word_map or src_word not in all_similarity_scores:
            print(("Dictionary doesn't contain " + unicode(src_word) + " . Can't compute MRR for the word\n").encode('utf-8'))
            continue

        target_word_sim_mapping = {w: all_similarity_scores[src_word][i] for w, i in target_word_index_mapping.items()}
        sorted_target_word_sim_tuples = sorted(target_word_sim_mapping.items(), key=lambda x: x[1], reverse=True)
        target_word_rank_mapping = {w[0]: i+1 for i, w in enumerate(sorted_target_word_sim_tuples)}

        # n.b. this code only really makes sense when there is one possible translation
        # i.e. the number of ranks and the MRR calculation denominator match
        for possible_translation in word_map[src_word]:
            if possible_translation not in target_word_rank_mapping:
                print(
                ("WARNING: " + possible_translation + " not in target_word_rank_mapping, skipping bad scenario").encode(
                    'utf-8'))
                continue

            # adds this translation pair to the set of results
            current_rank = target_word_rank_mapping[possible_translation]
            mrr_val_numerator += 1.0 / current_rank
            total_ranked_translation_pairs += 1
            all_ranks[src_word] = current_rank

    # output a singular warning message for any failed english words (avoids outputting once per loop)
    if len(failed_english_words) > 0:
        print(("target_word_index_mapping did not contain " + unicode(str(list(failed_english_words)))).encode('utf-8'))

    # Calculate Mean reciprocal rank
    mrr_val = mrr_val_numerator / total_ranked_translation_pairs

    return all_ranks, mrr_val

def get_feature_vector_from_qnt_file(qnt_img_file, key_type='sift'):
    if key_type == 'sift':
        def get_feat_vec_index(key):
            return int(key)
        feat_vec = np.zeros(100)
    else:
        def get_feat_vec_index(key):
            return histogram_hex_array_index_map[key]
        feat_vec = np.zeros(4096)

    try:
        file_contents = open(qnt_img_file).read().strip()
    except IOError as e:
        print "Error opening image feature vec files", e
        return

    if len(file_contents) == 0:
        print "Feat vec file can't be EMPTY"
        return

    # Transform the feat from the first feat file
    # to a list of features with feat key as the index
    for feat in file_contents.split(' '):
        feat_token = feat.split(':')
        feat_vec[get_feat_vec_index(feat_token[0])] = float(feat_token[1])

    return feat_vec


def form_dict(dict_path, possible_foreign_words, possible_english_words):
    """
        Forms a dictionary from a file with tab
        delimited translations with multiple src
        words. Format:
        <translation>\t<src_w1>\t<src_w2>\t...\t<src_wn>

    :param dict_path:
    :return:
    """
    # Open the dictionary file
    word_map = defaultdict(list)
    with io.open(dict_path, 'r', encoding='utf-8') as dict_f:
        # Form an in-memory map of true src-target
        # language mappings using the dictionary
        for line in dict_f.readlines():
            all = line.split("\t")  # Dict words separated by tabs
            foreign_word = all[0].strip()
            if foreign_word in possible_foreign_words:
                for english_word in all[1:]:
                    english_word = english_word.strip()
                    if english_word in possible_english_words:
                        word_map[foreign_word].append(english_word)

    return word_map

def compute_avg_max(similarities):
    """
        Method to compute the AVG-MAX similarity score
        for a given src word img set and target img
        set. Computes similarity score for every img
        in src and averages it.
    """
    return np.mean(np.max(similarities, axis=1))

# defines the compute sim score threads so we can generate results in a concurrent way
class ComputeSimScoreThread(threading.Thread):
    def __init__(self, src_word, src_feat_map, target_word_set, target_word_matrix_all, target_word_indices_map, all_similarity_scores, threadLimiter, lambdaa):
        self.src_word = src_word
        self.src_feat_map = src_feat_map
        self.target_word_set = target_word_set
        self.target_word_indices_map = target_word_indices_map
        self.target_word_matrix_all = target_word_matrix_all
        self.all_similarity_scores = all_similarity_scores
        self.threadLimiter = threadLimiter
        self.lambdaa = lambdaa
        threading.Thread.__init__(self)

    # simple wrapper function for thread safe console printing,
    # so we can wrap all of our multi-threaded console output in one function
    def thread_safe_print(self, output):
        with printing_lock:
            print(output.encode('utf-8'))

    def run(self):
        self.threadLimiter.acquire()
        try:
            self.thread_safe_print(u"Started finding matches for " + self.src_word)

            # also track all possibilities so we can output the top N options and their similarities
            # ordered in same way as target_word_set array
            options_for_current_word = []

            sift_features_src = self.src_feat_map["SIFT"][self.src_word]
            hist_features_src = self.src_feat_map["HIST"][self.src_word]

            has_src_sift_features = sift_features_src.shape[0] > 0 and sift_features_src.shape[1] > 0
            has_src_hist_features = hist_features_src.shape[0] > 0 and hist_features_src.shape[1] > 0

            if has_src_sift_features:
                all_cosine_similarities_sift = cosine_similarity(sift_features_src, self.target_word_matrix_all["SIFT"])
                all_cosine_similarities_sift = np.multiply(all_cosine_similarities_sift, self.lambdaa["SIFT"])

            if has_src_hist_features:
                all_cosine_similarities_hist = cosine_similarity(hist_features_src, self.target_word_matrix_all["HIST"])
                all_cosine_similarities_hist = np.multiply(all_cosine_similarities_hist, self.lambdaa["HIST"])

            # Computing similarity value for all candidate translations in target lang
            for target_word in self.target_word_set:
                sim_val = 0.0
                if has_src_sift_features and target_word in self.target_word_indices_map["SIFT"]:
                    sift_feature_indices = self.target_word_indices_map["SIFT"][target_word]
                    sift_similarities = all_cosine_similarities_sift[:,sift_feature_indices]
                    sim_val += compute_avg_max(sift_similarities)

                if has_src_hist_features and target_word in self.target_word_indices_map["HIST"]:
                    hist_feature_indices = self.target_word_indices_map["HIST"][target_word]
                    hist_similarities = all_cosine_similarities_hist[:,hist_feature_indices]
                    sim_val += compute_avg_max(hist_similarities)

                options_for_current_word.append(sim_val)

            if has_src_sift_features or has_src_hist_features:
                max_index, _ = max(enumerate(options_for_current_word), key=operator.itemgetter(1))
                max_target_word = self.target_word_set[max_index]
                self.thread_safe_print(u"src word - " + self.src_word + u" matches target word - " + max_target_word)
                self.all_similarity_scores[self.src_word] = options_for_current_word

        finally:
            self.threadLimiter.release()

def compute_sim_score(src_word_set, target_word_set,
                      src_feat_map, target_feat_map, lambdaa,
                      threadLimiter):
    """
        Computes the similarity score of src words against
        target word image sets using multiple feature
        classes

    :param src_word_set: Set of src language words
    :param target_word_set: Set of target language words
    :param src_feat_map: Feature classes from src lang images
    :param target_feat_map: Feature classes from target lang images
    :param lambdaa: Weights for each feature class
    :param n: Number of top similarity values to print to screen

    :return: match_words: src word - target word match
    """

    # Traverse through all source words and find the
    # best match among translations
    all_similarity_scores = {}

    thread_list = []

    target_word_indices_map = {"SIFT": {}, "HIST": {}}
    target_word_sift_vectors = []
    target_word_hist_vectors = []
    current_sift_index = 0
    current_hist_index = 0
    for target_word in target_word_set:
        if target_word in target_feat_map["SIFT"]:
            sift_features = target_feat_map["SIFT"][target_word]
            has_target_sift_features = sift_features.shape[0] > 0 and sift_features.shape[1] > 0

            if has_target_sift_features:
                next_sift_index = current_sift_index + sift_features.shape[0]
                target_word_sift_vectors.append(sift_features)
                target_word_indices_map["SIFT"][target_word] = range(current_sift_index, next_sift_index)
                current_sift_index = next_sift_index

        if target_word in target_feat_map["HIST"]:
            hist_features = target_feat_map["HIST"][target_word]
            has_target_hist_features = hist_features.shape[0] > 0 and hist_features.shape[1] > 0

            if has_target_hist_features:
                next_hist_index = current_hist_index + hist_features.shape[0]
                target_word_hist_vectors.append(hist_features)
                target_word_indices_map["HIST"][target_word] = range(current_hist_index, next_hist_index)
                current_hist_index = next_hist_index

    print("Running vstack for " + str(len(target_word_hist_vectors)) + " sets of hist vectors and "
          + str(len(target_word_sift_vectors)) + " sets of sift vectors")
    target_word_matrix_all = {
        "SIFT": sparse.vstack(target_word_sift_vectors),
        "HIST": sparse.vstack(target_word_hist_vectors)
    }

    # n.b. sanity check
    # for target_word in target_word_set:
    #     sift_indices = target_word_indices_map["SIFT"][target_word]
    #     hist_indices = target_word_indices_map["HIST"][target_word]
    #     sift_rebuild_test = target_word_matrix_all["SIFT"][sift_indices,:]
    #     hist_rebuild_test = target_word_matrix_all["HIST"][hist_indices,:]
    #     print(sift_rebuild_test.shape)
    #     print(hist_rebuild_test.shape)

    for src_word in src_word_set:
        current_thread = ComputeSimScoreThread(src_word, src_feat_map, target_word_set, target_word_matrix_all, target_word_indices_map,
                                               all_similarity_scores, threadLimiter, lambdaa)
        thread_list.append(current_thread)

    print("Starting " + str(len(thread_list)) + " jobs for computing similarity scores")

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    return all_similarity_scores

def create_filtered_path_list(word_feature_path, metadata_filepath, image_limit=None, language_map=None):
    word_dir_index = os.path.basename(metadata_filepath).split('.')[0]
    paths = []
    lang_data = []
    if metadata_filepath:
        metadata = json.load(open(metadata_filepath))
    sorted_prefiltered_list = [f for f in sorted(os.listdir(word_feature_path)) if f not in word_skip_files and f.endswith('.qnt')]

    for i, feature_filename in enumerate(sorted_prefiltered_list):
        if image_limit is not None and i >= image_limit:
            break
        feature_index = feature_filename.split('.')[0]
        if language_map:
            if 'ru' not in metadata[feature_index]['google']:
                lang_data.append((word_dir_index, feature_index, 'NotFound', 'Missing'))
                continue
            referring_url = metadata[feature_index]['google']['ru']
            if referring_url not in language_map:
                lang_data.append((word_dir_index, feature_index, 'NotFound', referring_url))
                continue
            elif not language_map[referring_url]:
                lang_data.append((word_dir_index, feature_index, 'WrongLang', referring_url))
                continue
            else:
                lang_data.append((word_dir_index, feature_index, 'InLang', referring_url))
        paths.append(feature_filename)
    return paths, lang_data


# defines the form image set threads so we can build our data set in a concurrent way
class FormFeatureSetThread(threading.Thread):
    def __init__(self, features_dir, features_dir_qnt, word_dir, image_limit, feature_sets_sift, feature_sets_hist,
                 threadLimiter, word=None, src_dir=None, url_language_map=None, lang_results=None, metadata_filepath=None):
        self.src_dir = src_dir
        self.features_dir = features_dir
        self.features_dir_qnt = features_dir_qnt
        self.word_dir = word_dir
        self.image_limit = image_limit
        self.feature_sets_sift = feature_sets_sift
        self.feature_sets_hist = feature_sets_hist
        self.threadLimiter = threadLimiter
        self.word = word
        self.url_language_map = url_language_map
        self.lang_results = lang_results
        self.metadata_filepath = metadata_filepath
        threading.Thread.__init__(self)

    # simple wrapper function for thread safe console printing,
    # so we can wrap all of our multi-threaded console output in one function
    def thread_safe_print(self, output):
        with printing_lock:
            print(output.encode('utf-8'))

    def run(self):
        self.threadLimiter.acquire()
        try:
            features_word_path_hist = self.features_dir + self.word_dir + "_hist"
            features_word_path_sift = self.features_dir_qnt + self.word_dir + "_sift_qnt"

            if self.word is None:
                src_word_path = self.src_dir + self.word_dir
                with io.open(src_word_path + "/word.txt", 'r', encoding='utf-8') as f:
                    word = f.read().strip()
            else:
                word = self.word

            self.thread_safe_print(u"Forming image set for " + word)

            # n.b. make sure we sort the output files so image_limit cuts off after the top N
            if self.metadata_filepath:
                paths, lang_data = create_filtered_path_list(features_word_path_sift, self.metadata_filepath,
                                                             self.image_limit, self.url_language_map)
                self.lang_results.append(lang_data)
                all_files_sift = paths

            else:
                all_files_sift = sorted([f for f in os.listdir(features_word_path_sift) if f not in word_skip_files])

            all_feats_sift = []
            for i, feat_file in enumerate(all_files_sift):
                # enforce a per-word image limit if configured
                if self.image_limit and i+1 >= self.image_limit:
                    break

                sift_feats = get_feature_vector_from_qnt_file(features_word_path_sift + "/" + feat_file, 'sift')
                all_feats_sift.append(sift_feats)
            self.feature_sets_sift[word] = sparse.csr_matrix(np.array(all_feats_sift))

            # n.b. make sure we sort the output files so image_limit cuts off after the top N
            if self.metadata_filepath:
                paths, lang_data = create_filtered_path_list(features_word_path_hist, self.metadata_filepath,
                                                             self.image_limit, self.url_language_map)
                #self.lang_results.append(lang_data) # n.b. only do this once for sift, as hist is identical
                all_files_hist = paths

            else:
                all_files_hist = sorted([f for f in os.listdir(features_word_path_hist) if f not in word_skip_files and f.endswith('.qnt')])

            all_feats_hist = []
            for i, feat_file in enumerate(all_files_hist):
                # enforce a per-word image limit if configured
                if self.image_limit and i+1 >= self.image_limit:
                    break

                hist_feats = get_feature_vector_from_qnt_file(features_word_path_hist + "/" + feat_file, 'hist')
                all_feats_hist.append(hist_feats)
            self.feature_sets_hist[word] = sparse.csr_matrix(np.array(all_feats_hist))

        finally:
            self.threadLimiter.release()

def form_image_sets(src_dir, features_dir, word_limit, image_limit, threadLimiter, url_language_map, lang_results, base_metadata_filepath):
    """
        Takes an image directory and forms sets
        of image path
    :param img_dir: Input dir containing images
    :return:
    """
    feature_sets_sift = {}
    feature_sets_hist = {}
    thread_list = []

    for word_dir in os.listdir(src_dir):
        if word_limit and len(thread_list) >= word_limit:
            break

        if word_dir == 'all_errors.json':
            continue

        metadata_filepath = base_metadata_filepath + '/' + word_dir + '.json' if base_metadata_filepath else ''

        current_thread = FormFeatureSetThread(features_dir, features_dir, word_dir, image_limit, feature_sets_sift,
                                              feature_sets_hist, threadLimiter, src_dir=src_dir,
                                              url_language_map=url_language_map, lang_results=lang_results, metadata_filepath=metadata_filepath)
        thread_list.append(current_thread)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()


    return {"SIFT": feature_sets_sift, "HIST": feature_sets_hist}

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

def form_image_sets_english(english_path_suffix_map, base_target_features_dir, base_target_qnt_features_dir, image_limit, threadLimiter):
    """
        Takes an image directory and forms sets
        of image path
    :param img_dir: Input dir containing images
    :return:
    """
    feature_sets_sift = {}
    feature_sets_hist = {}
    thread_list = []

    # N.B. the word limit does not apply here since it has already been enforced, and words can have multiple translations
    for word, word_dir_suffix in english_path_suffix_map.items():
        current_thread = FormFeatureSetThread(base_target_features_dir, base_target_qnt_features_dir, word_dir_suffix, image_limit, feature_sets_sift,
                                              feature_sets_hist, threadLimiter, word=word)
        thread_list.append(current_thread)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    return {"SIFT": feature_sets_sift, "HIST": feature_sets_hist}

def get_english_words_from_dict(foreign_dictionary_file, possible_foreign_words=None):
    english_words = set()
    with io.open(foreign_dictionary_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cells = line.split("\t")
            foreign_word = cells[0].strip()
            line_eng_words = cells[1:]
            if possible_foreign_words is None or foreign_word in possible_foreign_words:
                for word in line_eng_words:
                    english_words.add(word.strip())
    return english_words

def main(opts):
    threadLimiter = threading.BoundedSemaphore(opts.num_threads)
    threadLimiterCosineSim = threading.BoundedSemaphore(opts.num_threads_cosine_sim)

    src_dir = opts.src
    if not src_dir.endswith('/'):
        src_dir += '/'

    foreign_language_name = src_dir.split('/')[-2]

    base_target_dir = opts.english
    if not base_target_dir.endswith('/'):
        base_target_dir += '/'

    src_features_dir = opts.src_features
    if not src_features_dir.endswith('/'):
        src_features_dir += '/'

    base_target_features_dir = opts.english_features
    if not base_target_features_dir.endswith('/'):
        base_target_features_dir += '/'

    base_target_qnt_features_dir = opts.english_qnt_features
    if not base_target_qnt_features_dir.endswith('/'):
        base_target_qnt_features_dir += '/'

    # directory of dictionaries in multilingual-google-image-scraper repo
    dictionaries_dir = opts.dictionaries_dir
    if not dictionaries_dir.endswith('/'):
        dictionaries_dir += '/'

    output_dir = opts.output_dir
    if not output_dir.endswith('/'):
        output_dir += '/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    with io.open(os.path.dirname(os.path.realpath(__file__)) + "/language-code-map.json", 'r', encoding='utf-8') as f:
        language_code_map = json.load(f)
    dictionary_extension = language_code_map[foreign_language_name]
    foreign_dictionary_file = dictionaries_dir + "dict." + dictionary_extension

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

    lang_results = []

    print "Creating feature sets for foreign language\n"
    # Traverse SIFT feature vector files
    # and form src word image sets
    src_feature_sets = form_image_sets(src_dir, src_features_dir, opts.word_limit, opts.image_limit, threadLimiter, url_language_map, lang_results, opts.metadata_filepath)
    src_feature_sets_sift = src_feature_sets["SIFT"]
    src_feature_sets_hist = src_feature_sets["HIST"]

    src_word_set = list(set(src_feature_sets_sift.keys() + src_feature_sets_hist.keys()))

    possible_foreign_words = set(src_word_set)
    all_english_words = get_english_words_from_dict(foreign_dictionary_file, possible_foreign_words)
    english_path_suffix_map = get_english_path_suffixes(dictionaries_dir, all_english_words)

    print "Creating feature sets for english language\n"
    # Traverse SIFT feature vector files
    # and form target word image sets
    target_feature_sets = form_image_sets_english(english_path_suffix_map, base_target_features_dir,
                                                  base_target_qnt_features_dir, opts.image_limit, threadLimiter)
    target_feature_sets_sift = target_feature_sets["SIFT"]
    target_feature_sets_hist = target_feature_sets["HIST"]

    # Map defining different feature classes
    # for src language img features
    src_feat_map = {
        "SIFT": src_feature_sets_sift,
        "HIST": src_feature_sets_hist
    }

    # Map defining different feature classes
    # for target language img features
    target_feat_map = {
        "SIFT": target_feature_sets_sift,
        "HIST": target_feature_sets_hist
    }

    # Weights for each feature class
    lambdaa = {
        "SIFT": 1.0,
        "HIST": 0.5
    }

    target_word_set = list(set(target_feature_sets_sift.keys() + target_feature_sets_hist.keys()))

    target_word_index_mapping = {}
    for i, target_word in enumerate(target_word_set):
        target_word_index_mapping[target_word] = i

    print "Computing sim score for....\n"
    all_sim_scores = compute_sim_score(src_word_set, target_word_set, src_feat_map, target_feat_map, lambdaa, threadLimiterCosineSim)

    with io.open(output_dir + "all_sim_scores.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(all_sim_scores, ensure_ascii=False)))

    # FIXME: needs to be re-written to account for changed all_sim_scores format (key -> ordered array) instead of (key -> dict)
    if DEBUG:
        # get the top N options for this word and print them out
        top_n = 25
        for word, options_for_current_word in all_sim_scores.items():
            top_n_values = sorted(options_for_current_word.items(), key=operator.itemgetter(1),
                                  reverse=True)[0:top_n]
            print("----" + word + "----")
            for i, word_tuple in enumerate(top_n_values):
                word_option, word_sim = word_tuple
                print(str(i) + ':' + word_option + ":" + str(word_sim))

    print "Computing ranks....\n"
    # create a word mapping from the foreign words to their english translations,
    # limit the mapping to only include words that are in our list of scraped words
    possible_foreign_words = set(src_word_set)
    possible_english_words = set(target_word_set)
    word_map = form_dict(foreign_dictionary_file, possible_foreign_words, possible_english_words)

    # Compute MRR
    ranks, mrr_val = compute_MRR(src_word_set, word_map, all_sim_scores, target_word_index_mapping)
    print ranks
    print mrr_val

    output_dict = {
        'ranks': ranks,
        'mrr': mrr_val
    }

    # n.b. 100 / 10,000 serves as a standin for 5 / 500
    # as does 400 / 10,000 for 20 / 500
    accuracy_tests = [1, 5, 20, 100, 400]
    for top_n_val in accuracy_tests:

        top_n_accuracy = top_n_acc(top_n_val, ranks)
        print("Top " + str(top_n_val) + " results score is " + str(top_n_accuracy))
        output_dict['top_'+str(top_n_val)+'_accuracy'] = top_n_accuracy

    with io.open(output_dir + "target_word_set.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(target_word_set, ensure_ascii=False)))

    with io.open(output_dir + "results.json", 'w', encoding='utf-8') as json_file:
        json_file.write(unicode(json.dumps(output_dict, ensure_ascii=False)))

    with io.open(output_dir + "lang_analysis_results.tsv", 'w', encoding='utf-8') as tsv_file:
        tsv_file.write(u'\t'.join(('WORD', 'WORD_ID', u'IMG_INDEX', 'LANG_STATUS', 'REFERRING_URL', '\n')))
        for outer_tuple in lang_results:
          for inner_tuple in outer_tuple:
            tsv_file.write(u'\t'.join(inner_tuple + ('\n',)))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate Lexicon Induction Pipeline.')
    parser.add_argument('-l', '--word_limit', type=int, help='Word limit')
    parser.add_argument('-i', '--image_limit', type=int, help='Image limit')
    parser.add_argument('-d', '--dictionaries_dir', help='directory with image scraper repo dictionaries')
    parser.add_argument('-s', '--src', help='src dir')
    parser.add_argument('-e', '--english', help='target (english) base dir')
    parser.add_argument('-sf', '--src_features', help='src dir')
    parser.add_argument('-ef', '--english_features', help='target (english) base features dir')
    parser.add_argument('-efq', '--english_qnt_features', help='target (english) base features dir for sift qnt')
    parser.add_argument('-t', '--num_threads', default=64, type=int, help='number of threads to use')
    parser.add_argument('-tc', '--num_threads_cosine_sim', default=64, type=int, help='number of threads to use')
    parser.add_argument('-o', '--output_dir', help='output dir with result files')
    parser.add_argument('-mf', '--metadata_filepath',  default = '',
        help='google image crawl JSON metadata file (needed for language-confident eval)')
    parser.add_argument('-la', '--language_analysis_filepath', default = '',
        help='TSV of google:ru URL to is-in-language boolean (needed for language-confident eval)')
    opts = parser.parse_args()

    main(opts)
















