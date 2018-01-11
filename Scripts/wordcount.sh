#!bin/bash

#quick script to count number of words in pdf document (for latex!)
doc=$1
echo $doc
echo "counting words in $doc"
echo pdftotext $doc - | wc -c
