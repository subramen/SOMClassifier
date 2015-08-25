__author__ = 'surajman'
import numpy as np
from scipy import dot


def generate_positions(dimensions=[3,4]):
    result=[]
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            result.append([i,j])
    return result

# def euclidean(srcd=[random.random()*10 for i in range(300)], dst=[random.random()*10 for i in range(300)], dim=300):
#     res=0
#     for i in range(dim):
#         res = res+ (dst[i]-srcd[i])*(dst[i]-srcd[i])
#     return math.sqrt(res)


def numpydist(np1, np2):
    s=np.array(np1)
    t=np.array(np2)
    dist = np.linalg.norm(s-t)
    return dist

def cosdist(v1,v2):
    n1=np.array(v1)
    n2=np.array(v2)
    return dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)) # n1.n2/(|n1|.|n2|)
