#Change tagger to tree-tagger
#Use ndarrays to store tokens of each input document.


__author__ = 'surajman'
from collections import Counter
import json
import csv
import os
import re
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import POSTagger
import requests
from collections import OrderedDict, Counter
from sqlconn import SQLCon
from scipy.cluster.vq import kmeans2
import numpy


logging.basicConfig(level=logging.INFO, filename='/home/surajman/one.log')
logger = logging.getLogger(__name__)

def clean_tokenizer(out):
    out = re.sub(r'(\.)(^ )', r'\1 \2', out)  # Separates any 'done.Further' type non-spacing
    out = re.sub(r'\W+', ' ', out)  # Remove non words
    out = re.sub(r' x\w+', '', out)  # Remove encoding crap like \x94 etc
    out = re.sub(r'\d+', '', out)  # Remove digits
    out = re.sub(r' . ', ' ', out)  # Remove single chars
    tokens = [word for word in word_tokenize(out)[1:] if word.lower() not in stopwords.words('english')]
    # Starts from 1 to remove 'bytearray' prefix
    return tokens

def vectorizer(tokens, w2v_db):
    db_path = w2v_db
    # POS TAGGING
    tagger = POSTagger('tagger/english-left3words-distsim.tagger', 'tagger/stanford-postagger.jar')
    tagged_tokens = tagger.tag(tokens)
    unsorted_kw = OrderedDict()
    for (w,t) in tagged_tokens:
        if t in ['NNP', 'NNPS', 'FW']:
            label = 1.5
        elif t in ['NN', 'NNS']:
            label = 1
            
        else:
            continue
        w = w.lower()
        try:
            unsorted_kw[w] += label
        except KeyError:
            unsorted_kw[w] = label
    # Get the vectors of words. Maintain order as in document.
    token_vecs = OrderedDict()
    conn = SQLCon(db_path)
    words = (word.lower() for word in unsorted_kw)
    for word in words:
        try:
            if token_vecs[word]: continue
        except KeyError:
            v = conn.read(word)
            if not v is None:
                token_vecs[word] = list(v)
    print("kw_len: {0} vec_len: {1}".format(len(unsorted_kw), len(token_vecs))) #Output for debugging; total vs unique words.
    conn.close()
    return unsorted_kw, token_vecs

def clusterizer(token_vecs):
    #Compute cluster centers:
    nk = round(len(token_vecs)/4)
    data = numpy.array(list(token_vecs.values()))
    cent, _ = kmeans2(data,nk,iter=20,minit='points')
    centroids = cent.tolist()
    return centroids
    
def jsonizer(json_dir, jsonname, filepath, vecs, kw_freq, centroids):
    # Create the JSON object for this document.
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_path = os.path.join(json_dir,jsonname+'.json')
    file_dest = open(json_path, 'w')
    json.dump({'src_file': filepath, 'vectors' : vecs, 'keyword_frequency': kw_freq, 'centroids' : centroids}, file_dest)
    file_dest.close()


def csv_to_json(csvfile,db_path,extension='.txt'):
    json_dir = os.path.join(os.path.dirname(csvfile), 'json')
    flag = True
    try:
        existing_json_files = os.listdir(json_dir)
    except NotADirectoryError:
        flag = False
        pass
    f = open(csvfile,'r')
    reader = csv.reader(f)
    for row in reader:
        if len(row[2])<100: continue #100 is arbitrary
        id = int(row[0])
        category = row[1]
        tokens = clean_tokenizer(row[2])
        jsonname = category+'_'+id
        kw_freq, vecs = vectorizer(tokens, db_path)
        centroids = clusterizer(vecs)
        jsonizer(json_dir, jsonname, None, vecs, kw_freq, centroids):
        


def test():
    csv_to_json('/home/surajman/PycharmProjects/BEProj/20NG/b_a_20news.csv','/home/surajman/PycharmProjects/BEProj/w2v.db')


if __name__ == "__main__":
    test()
