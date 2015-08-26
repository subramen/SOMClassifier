# FOR LATER
# Work on dynamicism of word2vec. It should learn new terms/words. Use some lookup ontology.
# Phrase/MWE recognition
# Consider HDFS
# One search across multiple Map objects (Separation of 'search' as a class)
# Implement in cython/pypy

__author__ = 'surajman'
import random
import utilities
import math
import numpy as np
import logging
import json
import time
import os
import pickle
import webbrowser
from utilities import cosdist as distance
from utilities import numpydist
from urllib.request import pathname2url
from Neuron import Neuron
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import HoverTool
from sqlconn import SQLCon
from collections import Counter, OrderedDict
from scipy import dot
import jinja2


THIS_FOLDER = os.path.dirname(os.path.realpath(__file__))
ITERATIONS_MAX = 10
LEARNING_RATE_0 = 0.5

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO, filename='/home/surajman/one.log')
logger = logging.getLogger(__name__)

#jinja variables to render results.html
templateLoader = jinja2.FileSystemLoader( searchpath=THIS_FOLDER )
templateEnv = jinja2.Environment( loader=templateLoader )
TEMPLATE_FILE = "templates/template_v2.html"
template = templateEnv.get_template( TEMPLATE_FILE )

class Map:
    def __init__(self, map_dimensions=[15,15], weight_dimensions=200):
        self._map_dimensions = map_dimensions
        self._weight_dimensions = weight_dimensions
        self._neurons = []
        self._neu_index= {}
        self._glomat = {}

        # Initializing neurons with random weights:
        logger.info('Beginning init of neurons in map')
        for idx, position in enumerate(utilities.generate_positions(map_dimensions)):
            random_wt=[random.random()*10 for i in range(weight_dimensions)]
            self._neurons.append(Neuron(id=idx, position=position, weights=random_wt))
            self._neu_index.update({idx:self._neurons[-1]})
        logger.info('Map neurons initialized')

    # To find the best matching unit - neuron that is closest to input
    def find_bmu(self,input_vec):
        neurons = self._neurons
        wt_vec = neurons[0].get_weights()
        shortest_dist = distance(input_vec, wt_vec)
        bmu_neuron = None
        for i in range(len(neurons)):
            tmp_dist = distance(input_vec, neurons[i].get_weights())
            if tmp_dist >= shortest_dist:
                shortest_dist = tmp_dist
                bmu_neuron = neurons[i]
        return bmu_neuron

    # Modify the BMU-neighbourhood weights using exponential decay.
    def learn(self,dirname): #dirname = location of jsons
        neurons = self._neurons
        neu_len = len(neurons)
        epoch = 0
        l1 = neurons[int(neu_len/2)].get_position() # Middle neuron coordinate
        l2 = neurons[neu_len-1].get_position() # Last neuron coordinate
        radius0 = numpydist(l1,l2) # Initial radius (encompassing all nodes)
        time_const = ITERATIONS_MAX/radius0   #Lambda
        find_bmu = self.find_bmu
        toarray = np.array
        logger.info("Sigma0:%s TimeConst:%s", radius0, time_const)

        while epoch <= ITERATIONS_MAX:
            learning_rate = LEARNING_RATE_0 * math.exp(-(epoch/time_const))
            epoch_radius = radius0 * math.exp(-(epoch/time_const))
            # Iterate over JSONs
            filelist = os.listdir(dirname)
            nb_tot_files = len(filelist)
            random.shuffle (filelist)
            for idx,json_doc in enumerate(filelist):
                try:
                    with open(os.path.join(dirname,json_doc)) as f:
                        data = json.load(f)
                        vec_dict = data['vectors']
                except IsADirectoryError:
                    continue
                # Competition - Find BMU
                items = (w for w in vec_dict.items())
                for (word,vec) in items:
                    bmu_neuron = find_bmu(vec)
                    # Cooperation - Impact neighbourhood weight vectors
                    for neuron in neurons:
                        dist_from_BMU = numpydist(neuron.get_position(), bmu_neuron.get_position())
                        if dist_from_BMU < epoch_radius:        # Current neuron within BMU Radius
                            influence = learning_rate * math.exp(-(dist_from_BMU*dist_from_BMU)/(2*epoch_radius*epoch_radius))
                            current_wt = toarray(neuron.get_weights())
                            wt_diff = dot(influence,(toarray(vec) - current_wt))
                            new_wt = current_wt + wt_diff
                            neuron.set_weights(new_wt.tolist())
                remaining = nb_tot_files - idx - 1
                logger.info("Epoch %s  File:%s done.  %d files left.",epoch, json_doc, remaining)
            epoch+=1

    # Assigns doc file to appropriate neuron baed on cosine similarity to neuron weight-vector.
    def classify(self,dirname): # dirname = location of json
        filelist = os.listdir(dirname)
        for idx, json_doc in enumerate(filelist):
            this_file = os.path.join(dirname,json_doc)
            remaining = len(filelist) - idx
            logger.info("Classifying File:%s. %d files left.", json_doc,remaining)
            try:
                with open(this_file) as f:
                    file = json.load(f)
                    kw = Counter(file['keyword_frequency'])
                    top_kw = kw.most_common(round(len(kw)/10)) # top 10% (arbitrary) of keywords [(w,f),..]
                    vec_dict = file['vectors']
            except IsADirectoryError:
                continue
            neurons = self._neurons
            for neuron in neurons:
                doc_vec=0
                tot_freq=0
                for (w,f) in top_kw:
                    try:
                        doc_vec += dot(f, distance(vec_dict[w],neuron.get_weights())) # sum(freq*similarity(keyword,neuron))
                    except KeyError:
                        continue
                    tot_freq += f
                    neuron.write_keylist(w,distance(vec_dict[w], neuron.get_weights())) # {word:similarity_to_this_neuron}
                doc_vec /= tot_freq # weighted average of cosine similarity of all keywords in this_file to this neuron
                neuron.write_doclist(this_file,doc_vec)

    def draw_SOM (self,srch_str, dist_neu_searchq, neuron_labels,map_path):
        TOOLS = "pan,wheel_zoom,box_zoom,hover,resize"
        plot = figure(title=srch_str,title_text_font_size='20pt',tools=TOOLS,plot_width=1000,plot_height=1000)
        nid = sorted(dist_neu_searchq)
        dist = [dist_neu_searchq[n] for n in nid]
        norm_dist = [round((x-min(dist))*255/(max(dist)-min(dist))) for x in dist] # Light is might
        node_pos = utilities.generate_positions(self._map_dimensions)
        xx = [sub[0]for sub in node_pos]
        yy = [sub[1]for sub in node_pos]
        colours=["#%02x%02x%02x" % (g,0,g) for g in norm_dist]
        source = ColumnDataSource(
            data=dict(
                x=xx,
                y=yy,
                nid=nid,
                colors=colours,
                labels = neuron_labels
            )
        )
        plot.circle(xx,yy,source=source,size=50,fill_color=colours,fill_alpha=0.7,line_color=None)
        plot.text(xx, yy, text=norm_dist, alpha=0.5, text_font_size="15pt",text_baseline="middle", text_align="center")
        hover = plot.select(dict(type=HoverTool))
        hover.tooltips = [
            ("index", "@nid"),
            ("labels", "@labels")
        ]

        output_file(map_path)
        show(plot)



    # Returns unsorted dict of results {"Document" : "Score"}
    def search(self,search_str,search_dir,depth=2):
        # Depth -> number of top nodes to search. Default 2 (arbitrary) has been sufficient so far.
        results = Counter()        # {url:score}
        dist_neu_searchq=Counter()  # {nid:dist} Contains the similarity of each neuron to the aggregated search query
        neuron_lookup = self._neu_index
        neuron_labels = [[k for (k,n) in Counter(neuron.get_keylist()).most_common()[:10]] for neuron in self._neurons]
        glomat = self._glomat # Global matrix. Contains similarity of each search term in the query to all neurons. Something like a cache.

        conn = SQLCon()
        searchvecs = [(x,list(conn.read(x))) for x in search_str.split() if not conn.read(x) is None] # Obtain (word,vec) of search terms
        search_len = len(searchvecs)
        for (w,v) in searchvecs:        # For colour coding the map
            try:
                for nid in glomat[w]:
                    if glomat[w][nid] > dist_neu_searchq[nid]:
                        dist_neu_searchq[nid] += glomat[w][nid]/search_len
            except KeyError:
                glomat[w]={}
                for nid,neuron in enumerate(self._neurons):
                    glomat[w][nid] = distance(neuron.get_weights(),v) # cosine similarity, hence 1 is best. 0 is bleh. -1 is opposite.
                    if glomat[w][nid] > dist_neu_searchq[nid]:
                        dist_neu_searchq[nid] += glomat[w][nid]/search_len


            # Union of all doclists with minimum dist_from_neuron. 
            doclist = {}
            for nid in dist_neu_searchq.most_common()[:depth]:
                neuron = neuron_lookup[nid[0]]
                doclist.update(neuron.get_top_docs(30))

            files = (open(doc) for doc in doclist)
            for json_file in files:
                data = json.load(json_file)
                centroids = data['centroids']
                url = data['url']
                json_file.close()
                wc_sim = [distance(v,c) for c in centroids]
                max_wc_sim = max(wc_sim)
                results[url] += max_wc_sim/len(searchvecs)

        results = OrderedDict(results.most_common(20))
        htmlVars = {'query': search_str, 'results':results}
        htmlCode = template.render(htmlVars)
        result_path = os.path.join(search_dir,search_str+'.html')
        map_path = os.path.join(search_dir,search_str+'_map.html')

        with open(result_path,'w') as f:
            f.write(htmlCode)
        self.draw_SOM(search_str,dist_neu_searchq,neuron_labels,map_path)


        result_path = "file://{}".format(pathname2url(result_path))
        map_path = "file://{}".format(pathname2url(map_path))
        webbrowser.open(result_path)



def test():
    with open('/home/surajman/PycharmProjects/BEProj/docsouth/test/maps/ds_morelearn','rb') as f:
        algo = pickle.load(f)
    algo.search('dark eerie mystery ghost','/home/surajman')
    # for n in algo._neurons:
    #     pprint(n.get_data())

#dirname=destination_dir    pkl_name='save as' name of this map
def train(dirname,pkl_name):
    json_dir = os.path.join(dirname,'json')
    pkl_dir = os.path.join(dirname,'maps')
    pkl_path = os.path.join(pkl_dir,pkl_name)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    obj = Map()
    obj.learn(json_dir)
    op = open(pkl_path,'rb')
    pickle.dump(obj,op,pickle.HIGHEST_PROTOCOL)
    op.close()
    obj = pickle.load(op)
    obj.classify(json_dir)
    op = open(pkl_path,'wb')
    pickle.dump(obj,op,pickle.HIGHEST_PROTOCOL)
    op.close()
    print("Done")

def search(search_str, map_path):
    with open(map_path,'rb') as f:
        algo = pickle.load(f)
    root_dir = os.path.dirname(os.path.dirname(map_path))
    searchdir = os.path.join(root_dir,'searches')
    if not os.path.exists(searchdir):
        os.makedirs(searchdir)
    algo.search(search_str,searchdir)
    with open(map_path, 'wb') as f:
        pickle.dump(algo,f,pickle.HIGHEST_PROTOCOL) # Writing update glomat to object

if __name__ == '__main__':
    search('cloud computing', '/home/surajman/PycharmProjects/BEProj/acm/2014/maps/NSFAwards_trained_classified')
