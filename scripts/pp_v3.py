__author__ = 'surajman'
# preprocessor, 1) extracts webpage content, 2) POS tags it 3) Extracts NP chunks
# 4) Outputs JSON containg URL and Keyword dict

# To Do: Resolve excess imports

from collections import Counter
import json
import os
import re
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tag.stanford import POSTagger
import justext
import requests
from collections import OrderedDict, Counter
from sqlconn import SQLCon
from scipy.cluster.vq import kmeans2
import numpy


logging.basicConfig(level=logging.INFO, filename='/home/surajman/one.log')
logger = logging.getLogger(__name__)

# from logging.config import fileConfig
# fileConfig('logging_config.ini')
# logger = logging.getLogger()

# INPUTS
# Database to read vectors from
# Directory of docs


def cleanup(out):
    out = re.sub(r'(\.)(^ )', r'\1 \2', out)  # Separates any done.Further type non-spacing
    out = re.sub(r'\W+', ' ', out)  # Remove non words
    out = re.sub(r' x\w+', '', out)  # Remove crap like \x94 etc
    out = re.sub(r'\d+', '', out)  # Remove digits
    out = re.sub(r' . ', ' ', out)  # Remove single chars
    tokens = [word for word in word_tokenize(out)[1:] if word.lower() not in stopwords.words('english')]
    # Starts from 1 to remove 'bytearray' prefix
    return tokens


def get_url(webpage):
    doctext = bytearray()
    try:
        response = requests.get(webpage)
    except requests.exceptions.MissingSchema:
        webpage = 'http://' + webpage
        response = requests.get(webpage)
    paragraphs = justext.justext(response.content, justext.get_stoplist('English'))
    for para in paragraphs:
        if not para.is_boilerplate:
            doctext.extend(para.text.encode('UTF-8'))
    return cleanup(str(doctext))


def get_doc_contents(filepath):
    contents = bytearray()
    with open(filepath,'rb') as f:
        paragraphs = justext.justext(f.read(), justext.get_stoplist('English'))
    for para in paragraphs:
        if not para.is_boilerplate:
            contents.extend(para.text.encode('UTF8'))
    return cleanup(str(contents))  # LIST OF CLEANED TOKENS

def get_txt_contents(filepath):
    with open(filepath,'r') as f:
        flag = False
        blk=[]
        for line in f:
            if line.startswith("Title"):
               blk.append("%s " % (line[14:]))
            if flag:
                blk.append(line)
            else:
                if line.strip() == "Abstract    :":
                    flag = True
    blk = ''.join(blk)
    return cleanup(blk)

def processor(name, url, tokens, db_path,json_dir, USE_TITLE_WORDS = False):
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

    # Get the vectors list
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
    print("kw_len: {0} vec_len: {1}".format(len(unsorted_kw), len(token_vecs)))
    conn.close()

    #Compute cluster centers:
    nk = round(len(token_vecs)/4)
    data = numpy.array(list(token_vecs.values()))
    cent, _ = kmeans2(data,nk,iter=20,minit='points')
    centroids = cent.tolist()

    # Create the JSON object for this webpage.

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_path = os.path.join(json_dir,name+'.json')
    file_dest = open(json_path, 'w')
    json.dump({'url': url, 'vectors' : token_vecs, 'keyword_frequency': unsorted_kw, 'centroids' : centroids}, file_dest)
    file_dest.close()


def get_all_jsons(dirname,db_path,extension='.txt'):
    json_dir = os.path.join(dirname, 'json')
    flag = True
    try:
        existing_json_files = os.listdir(json_dir)
    except NotADirectoryError:
        flag = False
        pass
    for root, dirs, files in os.walk(dirname):
        l = len(files)
        for idx,filename in enumerate(files):
            if (flag and "{}.json".format(filename)) in existing_json_files:
                continue
            if not filename.endswith(extension):
                continue
            filepath = os.path.join(root,filename)
            logger.info("Preprocessing %s.",filename)
            try:
                toklist = get_doc_contents(filepath)
                l -= idx-1
            except IsADirectoryError:
                print("Not a file - %s" % filename)
                continue
            if len(toklist) < 10:
                continue
            else:
                processor(filename, filepath, toklist, db_path,json_dir)



def test():
    get_all_jsons('/home/surajman/PycharmProjects/BEProj/acm/Part 3/','/home/surajman/PycharmProjects/BEProj/w2v.db')
    # print(get_doc_contents('/home/surajman/websites/lread/web/videos.html'))


if __name__ == "__main__":
    test()