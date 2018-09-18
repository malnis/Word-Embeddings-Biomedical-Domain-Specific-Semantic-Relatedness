#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Kyle Tilbury, June 2018

# This program generates phrases

from __future__ import print_function

import logging
import os.path
import sys

from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath



if __name__ == '__main__':
    # Set up logging
    this_program = os.path.basename(sys.argv[0])
    log = logging.getLogger(this_program)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log.info("Running %s" % ' '.join(sys.argv))

    # Check arguments
    if len(sys.argv) < 3:
        print("Error. Run as \"PhraseGenerator <input text file> <output text file>")
        sys.exit(1)
    in_file, out_file = sys.argv[1:3]

    # Log input parameters
    log.info("Input file: " + in_file)
    log.info("Output file: " + out_file)

    # Apply phrase generator three times
    sentences = LineSentence(datapath(in_file))

    # First Pass
    log.info("Applying first phrasing pass")
    phrases_model = Phrases(sentences, max_vocab_size=200000000)
    bigram = Phraser(phrases_model)
    del phrases_model
    # Write phrased file
    log.info("Writing to output file")
    out = open(out_file + "1.txt", 'w+')
    for sentence in sentences:
        print( u' '.join(bigram[sentence]).encode('utf-8').strip(), file=out)
    out.close()

    # Second Pass
    log.info("Applying second phrasing pass")
    sentences = LineSentence(datapath(out_file + "1.txt"))
    phrases_model = Phrases(sentences, max_vocab_size=200000000)
    bigram = Phraser(phrases_model)
    del phrases_model
    # Write phrased file
    log.info("Writing to output file")
    out = open(out_file + "2.txt", 'w+')
    for sentence in sentences:
        print( u' '.join(bigram[sentence]).encode('utf-8').strip(), file=out)
    out.close()

    # Third Pass
    log.info("Applying third phrasing pass")
    sentences = LineSentence(datapath(out_file + "2.txt"))
    phrases_model = Phrases(sentences, max_vocab_size=200000000)
    bigram = Phraser(phrases_model)
    del phrases_model

    # Write phrased file
    log.info("Writing to output file")
    out = open(out_file +"3.txt", 'w+')
    for sentence in sentences:
        print( u' '.join(bigram[sentence]).encode('utf-8').strip(), file=out)
    out.close()
    log.info("Complete")