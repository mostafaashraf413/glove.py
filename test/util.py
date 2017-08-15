# -*- coding: utf-8 -*-
from functools import wraps
import re


def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified
    

# -*- coding: utf-8 -*-
#######################
#http://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
#as3:/usr/local/lib/python2.7/site-packages# cat sitecustomize.py
# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
########################
import gzip
import re

def extract_arabic_warc(warc_path):
    result_lst = []
    with gzip.open(warc_path, 'rb')as f:
        not_arChars = ur'[^ـ^\u0600-\u06FF^\u0750-\u077F^\u08A0-\u08FF^\uFB50-\uFDFF^\uFE70-\uFEFF]'
        counter = 0
        for line in f:
            line  = line.decode('utf-8')
            line = re.sub(not_arChars,u' ',line)
            line = re.sub(ur'\s+',u' ',line)
            line = line.strip()
            if len(line) > 0:
                result_lst.append(line)
                #print line
            counter+=1
            if counter % 100000 == 0:
                print 'reading...'
                
            if len(result_lst) == 1000:
                break
            
    return result_lst


def read_txt_file(file_path):
    result_lst = []
    with open(file_path, 'r')as f:
        for line in f:
            line = line.lower().strip()
            line = re.sub('[^a-z -]+', '', line)
            if len(line) > 1:
                result_lst.append(line)
    return result_lst
    
    
#x = read_txt_file('../resources/test_corpus.txt')
    
    
    
    
    
    
    
    