#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Kyle Tilbury, June 2018

# This program takes a a directory of full text OAS data and puts it into a single text file

import logging
import os.path
import sys
import glob
import shutil


if __name__ == '__main__':
    # Set up logging
    this_program = os.path.basename(sys.argv[0])
    log = logging.getLogger(this_program)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log.info("Running %s" % ' '.join(sys.argv))

    directory = "/home/kgt/workspace/train_data/oas/"

    count = 0
    # Loop through text files, appending to output file
    with open(directory + 'oas_raw.txt', 'wb') as wfd:
        for f in glob.glob(directory + "*/*.txt"):
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
                count += 1
                if 0 == count%100:
                    log.info("Files completed: "+str(count))

    log.info("Complete")