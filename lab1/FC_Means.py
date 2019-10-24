from random import sample, random, randint
from operator import itemgetter
from math import inf
import numpy as NP
from numpy import linalg as LA
from scipy.spatial.distance import pdist, euclidean
from scipy.io import loadmat
# from itertools import 
import matplotlib.pyplot as plt

def fuzzy_membership(x,clus,m):
  """returns the membership degree of the point x to the c centroids"""
  m_U = NP.zeros(len(clus))
  n_clus = len(clus)
  # d_down = sum([LA.norm((x-C)) for C in clus])
  d_down = sum([LA.norm((x-C)) for C in clus])
  for c in range(n_clus):
    if (x == clus[c]).all():
      # print(x,clus[c])
      m_U[c] = 1
    else:
      d_up = LA.norm(x-clus[c])
      d_b = (d_up/d_down)
      factor = (2.0/(m-1.0))
      d = d_b**factor
      d_f = 1.0/d
      m_U[c] = d_f

  return m_U/sum(m_U)

def my_KMeans(arr, n_clusters = 2, iter_m = 100):
  # # TODO randomly select K points as centroids
  n = len(arr)
  U = NP.zeros((n,n_clusters))
  idx = NP.zeros(n)
  centroids = arr[sample(range(n),K)]
  
  epoch = []
  # epoch.append(sum(W))

  changes = True
  old_idx = []
  iter = 0
  # centroids = NP.zeros((n_clusters,2))
  # TODO iterate until the cluster assignments stop changing
  while changes and iter<iter_m:
    # TODO assign each pattern to the cluster whose centroid is closest
    U = NP.zeros((n,n_clusters))
    idx = NP.zeros(n)
    for i in range(n):
      smallIdx = inf
      pattern = arr[i]
      smallDist = inf
      for j in range(K):
        gc = centroids[j]
        distance = LA.norm(pattern-gc)
        if distance<smallDist:
          smallDist = distance
          smallIdx = j
      U[i,smallIdx] = 1
      idx[i] = smallIdx

    # TODO compute the centroids
    centroids = NP.zeros((n_clusters,2))
    for j in range(n_clusters):
      onesIndx = NP.where(idx == j)
      Xj = arr[onesIndx]
      Uj = U[onesIndx]
      centroids[j] = NP.average(Xj,axis=0,weights=Uj.T[j])
    

    # TODO calculating the objective function
    clus = NP.unique(idx)
    c = len(clus)
    W = NP.zeros(c)
    for j in range(c):
      indxs = NP.where(idx == clus[j])
      Clusj = arr[indxs]
      # distance = [LA.norm(x[0]-x[1]) for x in combinations(Clusj,2)]
      distance = pdist(Clusj,'euclidean')
      W[j] = (1.0/float(len(indxs)))*sum(distance)
    
    iter += 1
    epoch.append(sum(W))

    # TODO verify stop criteria
    if NP.array_equal(idx,old_idx):
      changes = False
    else:
      old_idx = idx

  return epoch, idx, centroids, iter

def my_FCMeans(arr, n_clusters = 2, iter_m = 100, m = 2, criteria=0.01):
  # TODO randomly select K points as centroids
  n = len(arr)
  U = NP.zeros((n,n_clusters))
  idx = NP.zeros(n)
  centroids = arr[sample(range(n),n_clusters)]
  
  epoch = []
  changes = True
  old_idx = idx[:]
  iter = 0

  # TODO iterate until the cluster assignments stop changing
  while changes and iter<iter_m:
    idx = NP.zeros(n)

    # TODO assign each pattern to the cluster whose centroid is closest with Fuzzy
    U = NP.zeros((n,n_clusters))
    for i in range(n):
      U[i] = fuzzy_membership(arr[i],centroids,m)
      idx[i] = NP.argmax(U[i])

    # TODO compute the centroids
    centroids = NP.zeros((n_clusters,2))
    for j in range(n_clusters):
      onesIndx = NP.where(idx == j)
      Xj = arr[onesIndx]
      Uj = U[onesIndx]
      centroids[j] = NP.average(Xj,axis=0,weights=Uj.T[j])
    
    # TODO calculating the objective function
    clus = NP.unique(idx)
    c = len(clus)
    W = NP.zeros(c)
    for j in range(c):
      indxs = NP.where(idx == clus[j])
      Clusj = arr[indxs]
      # distance = [LA.norm(x[0]-x[1]) for x in combinations(Clusj,2)]
      distance = pdist(Clusj,'euclidean')
      W[j] = (1.0/float(len(indxs)))*sum(distance)
    
    iter += 1
    print(iter)
    epoch.append(sum(W))

    # TODO verify stop criteria
    if iter >1 and abs(epoch[iter-1]-epoch[iter-2]) < criteria:
      changes = False
    else:
      old_idx = idx
    # if NP.array_equal(idx,old_idx):
    #   changes = False
    # else:
    #   old_idx = idx

  return epoch, idx, centroids, iter

if __name__ == "__main__":
  
  # TODO load data
  mat = loadmat('fcm_dataset.mat')

  # X = -2 * NP.random.rand(150,2)
  # X1 = 1 + 2 * NP.random.rand(50,2)
  # X2 = 3 + 5*NP.random.rand(50,2)
  # X[50:100, :] = X1
  # X[100:150, :] = X2

  K = 4
  std_d = 0.6

  # dist_1_x = NP.random.normal(2,std_d, 200)
  # dist_1_y = NP.random.normal(2,std_d, 200)
  # dist_2_x = NP.random.normal(4,std_d, 200)
  # dist_2_y = NP.random.normal(4,std_d, 200)

  # dist_1 = NP.array([dist_1_x,dist_1_y])
  # dist_1 = dist_1.T
  # dist_2 = NP.array([dist_2_x,dist_2_y])
  # dist_2 = dist_2.T

  # dist = NP.concatenate((dist_1,dist_2))
  # 
  # epc, indexes, centers, itera= my_FCMeans(mat['x'],K)
  epc = 0
  indexes = 0
  centers = 0
  itera = 0
  i = 1
  epc_chp = []
  while i > 0:
    try:
      epc, indexes, centers, itera= my_FCMeans(mat['x'],K)
      # epc, indexes, centers, itera = my_KMeans(mat['x'],K)
      print(itera)
      i -= 1
      epc_chp.append(epc)
    except ZeroDivisionError as err:
      print(err)

  # epc, indexes, centers,itera= my_FCMeans(X,K)
  # epc, indexes, centers, itera= my_FCMeans(dist,K)
  # print(itera)

  # with open('log.csv', 'w') as f:
  #     for item in epc_chp:
  #         f.write("%s\n" % NP.mean(item))

  plt.figure(1)
  # plt.scatter(*dist_1.T)
  # plt.scatter(*dist_2.T)

  plt.scatter(*mat['x'].T)
  plt.title('Clusterização com FCMeans - Dados iniciais')
  # plt.scatter(*X.T)
  plt.figure(2)
  for i in range(K):
    # dist_sca = X[NP.where(indexes == i)]
    dist_sca = mat['x'][NP.where(indexes == i)]
    # dist_sca = dist[NP.where(indexes == i)]
    plt.scatter(*dist_sca.T)
  # plt.figure(3)
  plt.scatter(*centers.T,c='k')
  plt.title('Clusterização com FCMeans - %i Clusters + Centros'%K)
  plt.show()
