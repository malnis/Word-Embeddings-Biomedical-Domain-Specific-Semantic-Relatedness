#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Kyle Tilbury, June 2018

# This program takes a a directory of Pubmed data, extracts title and abstract, and puts it into a single text file

import logging
import os.path
import sys
import glob
from xml.etree import cElementTree as ET


if __name__ == '__main__':
    # Set up logging
    this_program = os.path.basename(sys.argv[0])
    log = logging.getLogger(this_program)

    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log.info("Running %s" % ' '.join(sys.argv))



    directory = "/home/kgt/workspace/train_data/"

    count = 0
    # Loop through text files, appending to output file
    with open(directory + 'pubmed_raw.txt', 'wb') as wfd:
        for f in glob.glob(directory + "pubmed/*.xml"):
            xml_tree = ET.parse(f)
            xml_root = xml_tree.getroot()

            for article in xml_root:
                count += 1
                if 0 == count%100:
                    log.info("Articles completed: "+str(count))

                title = article.find('MedlineCitation').find('Article').find('ArticleTitle').text
                if title is not None:
                    wfd.write(title.encode('utf-8') + "\n")


                abstract = article.find('MedlineCitation').find('Article').find('Abstract')
                if abstract is not None:
                    abstract_text = abstract.find('AbstractText').text
                    if abstract_text is not None:
                        wfd.write(abstract_text.encode('utf-8') + "\n")

    log.info("Complete")
