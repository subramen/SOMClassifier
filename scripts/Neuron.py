__author__ = 'surajman'
from operator import itemgetter
from collections import Counter

class Neuron:
    def __init__(self, id=-1, position=[], weights=[]):
        self._position = position
        self._weights = weights #Can it be a np array?
        self._id = id
        self._doclist= {}
        self._keylist = {} # {term:tf}

    def get_id(self):
        return self._id

    # Filename is the path to doc JSON
    def write_doclist(self,filename,distance):
        self._doclist.update({filename:distance})

    def get_top_docs(self, n=0):
        if n==0:
            n = round(len(self._doclist)/10)
        return Counter(self._doclist).most_common(n)

    def get_keylist(self):      # Dict of keywords
        return self._keylist

    def write_keylist(self,word,dist):
            self._keylist.update({word:dist})

    def get_position(self):
        return self._position

    def set_position(self, position):
        self._position = position

    def get_weights(self):
        return self._weights

    def set_weights(self,weights):
        self._weights = weights


