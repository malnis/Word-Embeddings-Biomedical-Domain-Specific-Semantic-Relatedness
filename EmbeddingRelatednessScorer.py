#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Kyle Tilbury, June 2018

# This program takes a directory of relatedness evaluation dataset CSV files and a directory of word embeddings, computes
# the relatedness score between pairs of words in the eval sets for each embedding, then outputs a CSV for each input CSV
# of the form |Word 1|Word 2|Human (mean)|Embedding_1_Rel|Embedding_2_Rel|...


from __future__ import print_function

import logging
import os.path
import sys
import glob
import pandas as pd
import numpy as np

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from itertools import islice
from scipy import spatial


def get_multiterm_emb(word, vocab, model):

    words = word.split('_')
    if words[0] not in vocab:
        return None
    multi_term_emb = model[words[0]]
    for term in islice(words, 1, None):
        if term not in vocab:
            return None
        multi_term_emb = np.add(multi_term_emb, model[term])
    return multi_term_emb



if __name__ == '__main__':
    # Set up logging
    this_program = os.path.basename(sys.argv[0])
    log = logging.getLogger(this_program)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log.info("Running %s" % ' '.join(sys.argv))

    # Check arguments
    if len(sys.argv) < 3:
        print("Error. Run as \"EmbeddingRelatednessScorer <directory of csv datasets> <directory of sets of word embeddings> <output directory>")
        sys.exit(1)
    csv_directory, embedding_directory, output_directory = sys.argv[1:4]

    # Log input parameters
    log.info("Directory of csv datasets: " + csv_directory)
    log.info("Word embedding directory: " + embedding_directory)
    log.info("Output directory: " + output_directory)

    # Create dictionary of evaluation data sets
    eval_datasets_dict = {}
    log.info("Adding evaluation datasets...")
    # Loop through csv files, create dataframes, add to dict
    for csv_file in glob.glob(csv_directory + "*.csv"):
        # Read csv to dataframe
        df = pd.read_csv(csv_file)
        # Get dataset name
        eval_set_name = csv_file.split('/')[-1:][0].split('.')[0]
        # add dataset to dict with key = name of dataset, value = the dataframe containing it
        eval_datasets_dict[eval_set_name] = df
        log.info("Added " + eval_set_name)


    # Loop through word embeddings, evaluate relatedness, add model relatedness column to each datasets dataframe
    for model_file in glob.glob(embedding_directory + "*"):
        log.info("Model file: " + model_file)
        # Load model
        if "glove" in model_file:
            glove_file = datapath(model_file)
            tmp_file = get_tmpfile("temp_word2vec.txt")
            glove2word2vec(glove_file, tmp_file)
            model = KeyedVectors.load_word2vec_format(tmp_file)
        elif "bin" in model_file:
            log.info("Binary file")
            model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        else:
            model = KeyedVectors.load_word2vec_format(model_file)

        # Print model name
        # This next line might mess up depending on whether path uses '/' or '\'
        model_name = model_file.rpartition('/')[2].split('.')[0]

        output_file = open(output_directory + "extra/" + model_name + ".out", 'w+')
        print("Model name: " + model_name, file=output_file)

        # Get vocab to calculate out of vocab (OOV) words
        vocab = model.wv.vocab
        print("Vocab size: " + str(len(vocab)), file=output_file)

        # Print some similar
        # Words multiple senses
        print("Most similar to 'culture': ", file=output_file)
        print(model.wv.most_similar("culture", topn=30), file=output_file)
        print("Most similar to 'acid': ", file=output_file)
        print(model.wv.most_similar("acid", topn=30), file=output_file)
        print("_____________________________________________________________________________________", file=output_file)
        if 'polyp' in vocab:
            print("Most similar to 'polyp': ", file=output_file)
            print(model.wv.most_similar('polyp', topn=30), file=output_file)
        if 'antibiotic' in vocab:
            print("Most similar to 'antibiotic': ", file=output_file)
            print(model.wv.most_similar('antibiotic', topn=30), file=output_file)
        if 'prozac' in vocab:
            print("Most similar to 'prozac': ", file=output_file)
            print(model.wv.most_similar('prozac', topn=30), file=output_file)
        if 'cardiomyopathy' in vocab:
            print("Most similar to 'cardiomyopathy': ", file=output_file)
            print(model.wv.most_similar('cardiomyopathy', topn=30), file=output_file)
        print("_____________________________________________________________________________________", file=output_file)

        # Loop through evaluation datasets
        for eval_set_name, df in eval_datasets_dict.iteritems():
            log.info("Evaluating model on " + eval_set_name)
            print("Eval dataset: " + eval_set_name, file=output_file)
            sim = []
            oov = 0
            print("======================================", file=output_file)
            print("OOV terms: ", file=output_file)


            # Loop through each pair of words in this dataset and get the similarity
            for index, row in df.iterrows():
                word_1 = str(row['Word 1']).lower().replace('-', '_')
                word_2 = str(row['Word 2']).lower().replace('-', '_')

                word_1_in_vocab_bool = word_1 in vocab
                word_2_in_vocab_bool = word_2 in vocab
                # If both terms in vocab them get the cosine sim between their vectors
                if word_1_in_vocab_bool and word_2_in_vocab_bool:
                    sim.append(model.wv.similarity(word_1, word_2))

                # If word 1 not in vocab and word 2 is in
                if not word_1_in_vocab_bool and word_2_in_vocab_bool:
                    # If word 1 is a multiword term we could possible get embedding for it and then use it
                    if '_' in word_1:
                        multiterm_emb = get_multiterm_emb(word_1, vocab, model)
                        # If the multiterm embedding is None then some term within it was OOV, and this pair is OOV
                        if multiterm_emb is None:
                            sim.append(None)
                            oov += 1
                            print(word_1, file=output_file)
                        else:
                            sim.append(1 - spatial.distance.cosine(multiterm_emb, model[word_2]))
                    # Else if its not multiword then its simply OOV
                    else:
                        sim.append(None)
                        oov += 1
                        print(word_1, file=output_file)

                # If word 1 is in vocab and word 2 not in
                if word_1_in_vocab_bool and not word_2_in_vocab_bool:
                    # If word 2 is a multiword term we could possible get embedding for it and then use it
                    if '_' in word_2:
                        multiterm_emb = get_multiterm_emb(word_2, vocab, model)
                        # If the multiterm embedding is None then some term within it was OOV, and this pair is OOV
                        if multiterm_emb is None:
                            sim.append(None)
                            oov += 1
                            print(word_2, file=output_file)
                        else:
                            sim.append(1 - spatial.distance.cosine(model[word_1], multiterm_emb))
                    # Else if its not multiword then its simply OOV
                    else:
                        sim.append(None)
                        oov += 1
                        print(word_2, file=output_file)

                # If word 1 not in vocab and word 2 not in vocab
                if not word_1_in_vocab_bool and not word_2_in_vocab_bool:
                    word_1_is_multi_bool = '_' in word_1
                    word_2_is_multi_bool = '_' in word_2
                    # If both words are multiword term we could possible get embeddings for them
                    if word_1_is_multi_bool and word_2_is_multi_bool:
                        multiterm_emb_1 = get_multiterm_emb(word_1, vocab, model)
                        multiterm_emb_2 = get_multiterm_emb(word_2, vocab, model)
                        # if both terms have embeddings, aka are not none then get their sim
                        if multiterm_emb_1 is not None and multiterm_emb_2 is not None:
                            sim.append(1 - spatial.distance.cosine(multiterm_emb_1, multiterm_emb_2))
                        # Else if one of them are none then OOV
                        else:
                            sim.append(None)
                            oov += 1
                            if multiterm_emb_1 is None:
                                print(word_1, file=output_file)
                            if multiterm_emb_2 is None:
                                print(word_2, file=output_file)
                    # Else if both not multiword, then one (or both) are simply OOV
                    else:
                        sim.append(None)
                        oov += 1
                        # Print oov term(s) to file
                        if not word_1_is_multi_bool:
                            print(word_1, file=output_file)
                        if not word_2_is_multi_bool:
                            print(word_2, file=output_file)


            print("======================================", file=output_file)
            # print OOV stats to the file
            print("OOV count: " + str(oov), file=output_file)
            num_pairs = float(''.join(filter(str.isdigit, eval_set_name)))
            print( "OOV %: " + str(oov / num_pairs), file=output_file)
            print("_____________________________________________________________________________________",
                  file=output_file)

            # add the embedding models realtedness scores to the datasets dataframe
            df[model_name+'_Rel'] = pd.Series(sim).values

        output_file.close()

    # Save results to csvs
    log.info("Saving results to file")
    for eval_set_name, df in eval_datasets_dict.iteritems():
        df.to_csv(output_directory + eval_set_name + "_output.csv", index=False)

    log.info("Complete")

