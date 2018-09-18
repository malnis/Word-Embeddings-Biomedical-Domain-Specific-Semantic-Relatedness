#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Kyle Tilbury, June 2018

# This program takes a directory of CSV files of the form |Word 1|Word 2|Human (mean)|Embedding_1_Rel|Embedding_2_Rel|..
# and evaluates correlation between embedding scores and the human score

import logging
import os.path
import sys
import glob
import pandas as pd

# The methods used for similarity -- as named in the results CSVs
methods_all = [ 'glove-web_Rel',
                'fasttext-web_Rel',
                'fasttext-wikinews_Rel',
                'word2vec-pubmed-T1_Rel',
                'word2vec-pubmed-A1_Rel',
                'fasttext-pubmed_Rel',
                'fasttext-oas_Rel',
                'fasttext-pub+oas_Rel']


methods_bio = [ 'word2vec-pubmed-T1_Rel',
                'word2vec-pubmed-A1_Rel',
                'fasttext-pubmed_Rel',
                'fasttext-oas_Rel',
                'fasttext-pub+oas_Rel']

methods_full = ['fasttext-webE_Rel',
                'fasttext-wikinewsE_Rel',
                'fasttext-pubmedE_Rel',
                'fasttext-oasE_Rel',
                'fasttext-pub+oasE_Rel']

methods_pub = [ 'fasttext-pubmed_full_Rel',
                'fasttext-pubmed_fullSubword_Rel'
                ]



if __name__ == '__main__':
    # Set up logging
    this_program = os.path.basename(sys.argv[0])
    log = logging.getLogger(this_program)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log.info("Running %s" % ' '.join(sys.argv))

    # Check arguments
    if len(sys.argv) < 4:
        print("Error. Run as \"WordSimCorrelation <directory of csv results> <all, bio, full> <'p' for pearson"
              " correlation or 's' for spearman> <'z' for setting OOV sim to 0, 'e' for excluding OOV pairs when calced,"
              "or 'i' for only calculating correlation between pairs that have a sim score for all methods>\" ")
        sys.exit(1)
    csv_directory, methods_evaluated, corr, oov_handling = sys.argv[1:5]

    if methods_evaluated == 'all':
        methods = methods_all
    elif methods_evaluated == 'bio':
        methods = methods_bio
    elif methods_evaluated == 'full':
        methods = methods_full
    elif methods_evaluated == 'pub':
        methods = methods_pub
    else:
        print("Error. Methods specification.")
        sys.exit(1)

    if corr == 'p':
        corr_method = 'pearson'
    elif corr == 's':
        corr_method = 'spearman'
    else:
        print("Error. Correlation specification.")
        sys.exit(1)

    if oov_handling == 'z':
        print("OOV word pair similarity scores set to 0.")
    elif oov_handling == 'i':
        print("OOV word pair similarity scores kept only for pairs with a score from all methods.")
    elif oov_handling == 'e':
        print("OOV word pair similarity scores ignored in correlation calculation.")
    else:
        print("Error. OOV handling specification.")
        sys.exit(1)

    # Loop through csv files
    for csv_file in glob.glob(csv_directory + "*.csv"):
        df = pd.read_csv(csv_file)
        dset = csv_file.split('/')[-1:][0].split('.')[0]
        # Handle out of vocab (OOV) sim scores
        if oov_handling == 'z':
            df = df.fillna(0)
            print(dset)
        elif oov_handling == 'i':
            df = df.dropna()
            print(dset + "(" + str(df.shape[0]) + " pairs)")
        elif oov_handling == 'e':
            # Do nothing here?
            print(dset)
        #print df

        for m in methods:
            corr_result = df['Human (mean)'].corr(df[m], method=corr_method)
            if oov_handling == 'e':
                num_oov = len(df[m]) - df[m].count()
                percent_oov = num_oov / float(filter(str.isdigit, dset))
                print(m + "\t" + str(corr_result) + "\t" + str(percent_oov))
            else:
                print(m + "\t" + str(corr_result))
