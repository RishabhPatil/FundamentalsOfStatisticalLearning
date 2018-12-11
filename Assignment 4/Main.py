
# coding: utf-8

# In[1]:

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import time
# import pylab as pl
# from IPython import display
from scipy.stats import multivariate_normal
import re
# import random
np.random.seed(420)

# # Helper Functions

# In[2]:

def read_file(filename):
    with open(filename, "r") as f:
        data = f.read()[:-2]

    data = re.sub("^( )*","",data)
    data = re.sub("( )*\n( )*","\n",data)
    data = re.sub("( )+"," ",data)
    data_l = data.split("\n")
    for i in range(len(data_l)):
        data_l[i] = data_l[i].split(" ")
        data_l[i] = [ float(j) for j in data_l[i] ]
    data_l = np.array(data_l)
    return data_l

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def draw_loss_graph(losses, filepath):
    plt.clf()
    x_points = [i for i in range(0,len(losses))]
    cost_line, = plt.plot(x_points, losses)
    plt.title("Avg SSE v/s Iteration")
    plt.savefig(filepath)
#     plt.show()

def draw_logloss_graph(losses, filepath):
    plt.clf()
    x_points = [i for i in range(0,len(losses))]
    cost_line, = plt.plot(x_points, losses)
    plt.title("Avg LogLoss v/s Iteration")
    plt.savefig(filepath)
#     plt.show()


# # KMeans

# In[3]:

def init_centers(data, K):
    low, high = np.min(data, axis=0), np.max(data, axis=0)
    centers = []
    for i in range(K):
        centers.append(np.random.uniform(low, high))
    return np.array(centers)

def get_clusters(data, centers):
    diff = (data[:, np.newaxis] - centers)**2
    dist_sq = np.sum(diff, axis=2)
    pred = np.argmin(dist_sq, axis=1)
    sse = np.sum(np.min(dist_sq, axis=1))/data.shape[0]
    return pred, sse

def step(data, pred, K):
    pred_onehot = one_hot(pred, K)
    group_sum = np.sum(data[:, np.newaxis] * pred_onehot[:,:,np.newaxis], axis=0, keepdims=True).reshape(pred_onehot.shape[1],2)
    centroids = group_sum/np.sum(pred_onehot, axis=0).reshape(pred_onehot.shape[1],1)
    return centroids

def display_clusters(data, centers, im, filepath):
    plt.clf()
    cluster_colors = np.array([(1,0,0),(0,1,0),(0,0,1)])
    colors = one_hot(im, centers.shape[0]).dot(cluster_colors[:centers.shape[0]])
    plt.title(filepath)
    h1 = plt.scatter(data[:,0], data[:,1], c=colors, label="Data Points")
    h2 = plt.scatter(centers[:,0], centers[:,1], c='b', marker='x', label="Centers")
    plt.legend(handles=[h1,h2])
    plt.savefig(filepath+".png")
    # plt.show()

def gif(data, ims, c_array):
    plt.clf()
    for i in range(len(ims)):
        display.clear_output(wait=True)
        display.display(pl.gcf())
        centers = c_array[i]
        plt.scatter(data[:,0], data[:,1], c=ims[i], animated=True)
        plt.scatter(centers[:,0], centers[:,1], c='b', marker='x')
        time.sleep(0.1)
    x = plt.clf()
    
def kmeans(data, K, verbose = True):
    centers = init_centers(data, K)
    prev = np.zeros((K,2))
    preds = []
    c_array = []

    pred, sse = get_clusters(data, centers)
    stdouts = "Initial Loss: "+str(sse)+"\n"
    losses = [sse]

    for i in range(100):
        centers = step(data, pred, K)
        pred, sse = get_clusters(data, centers)
        if verbose:
            stdouts += "Iteration: "+str(i+1)+" Loss: "+str(sse)+"\n"
        if (prev == centers).all():
            break
        prev = centers
        losses.append(sse)
        preds.append(pred)
        c_array.append(centers)
    
    if verbose:
        print(stdouts)
    return preds[-1], centers , losses


# # Gaussian Mixture Models

# In[4]:

def random_init_gmm(data, K):
    low, high = np.min(data, axis=0), np.max(data, axis=0)
    centers = []
    sigmas = []
    pis = np.random.rand(K,1)
    pis = pis/np.sum(pis)
    for i in range(K):
        centers.append(np.random.uniform(low, high))
        sigmas.append(np.cov(data.T))

    return np.array(centers), np.array(sigmas), pis

def init_gmm(data, c, preds):
    centers = []
    sigmas = []
    pis = np.random.rand(K,1)
    pis = pis/np.sum(pis)
    for i in range(c.shape[0]):
        ix = np.where(preds==i)
        centers.append(c[i])
        sigmas.append(np.cov(data[ix].T))

    return np.array(centers), np.array(sigmas), pis

def get_pdfs(mus, sigmas):
    pdfs = []
    for i in range(len(mus)):
        pdfs.append(multivariate_normal(mus[i], sigmas[i]).pdf)
    return pdfs

def get_piN(data, pdfs, pis, K):
    N = np.zeros((data.shape[0], K)) #*
    for i in range(K):
        N[:,i] = np.apply_along_axis(pdfs[i],1,data)
    return N*pis.T

def get_resp(piN):
    s = np.sum(piN, axis=1, keepdims=True)
    resp = piN/s
    return resp

def m_step(data, resp):
    n_k = np.sum(resp, axis=0)
    
    new_mus = np.sum(data[:,np.newaxis] * resp[:,:,np.newaxis], axis=0) / n_k[:,np.newaxis]
     
    diff = (data[:,np.newaxis] - new_mus)
    x = resp[:,:,np.newaxis,np.newaxis]*(diff.reshape(data.shape[0],resp.shape[1],2,1)*diff.reshape(data.shape[0],resp.shape[1],1,2))
    new_sigmas = np.sum(x,axis=0)/n_k[:,np.newaxis,np.newaxis]
    
    new_pis = n_k/data.shape[0]
    
    return new_mus, new_sigmas, new_pis

def loglikelihood(piN):
    x = np.sum(piN, axis = 1)
    x = np.sum(np.log(x))
    return -x/piN.shape[0]

def display_gmm(data, resp, K, filepath):
    plt.clf()
    cluster_colors = [(1,0,0),(0,1,0),(0,0,1)]
    colors = resp.dot(cluster_colors[:K])
    plt.title(filepath)
    plt.scatter(data[:,0], data[:,1], c=colors)
    plt.savefig(filepath+".png")
    # plt.show()


# # Running for Dataset_1.txt

# In[5]:

data = read_file("Dataset_1.txt")
K = 2


# ### KMeans

# In[6]:

r = 5
loss_array = []
preds_array = []
centers_array = []

for i in range(5):
    print("Run number",i+1)
    preds, centers, losses = kmeans(data, K)
    loss_array.append(losses[-1])
    preds_array.append(preds)
    centers_array.append(centers)
    draw_loss_graph(losses, "KMeans_Dataset1_Run"+str(i)+"_SSE.png")
    
ix = np.argmin(loss_array)
print("Lowest SSE during Run",ix,"SSE=",loss_array[ix])
display_clusters(data, centers_array[ix], preds_array[ix], "KMeans_Dataset1_Run"+str(ix))


# ### Random Init GMM

# In[7]:

#160,566,754,630
# i = int(random.random()*1000)
# print(i)
# np.random.seed(i)

mus, sigmas, pis = random_init_gmm(data, K)
pdfs = get_pdfs(mus, sigmas)
piN = get_piN(data, pdfs, pis, K)
prev = loglikelihood(piN)
print("Initial Avg. LogLoss:",str(prev))
log_losses = [prev]

i=0
while True:
    resp = get_resp(piN)
    mus, sigmas, pis = m_step(data, resp)
    pdfs = get_pdfs(mus, sigmas)
    piN = get_piN(data, pdfs, pis, K)
    x = loglikelihood(piN)
    print("Iteration "+str(i+1)+" Avg. LogLoss: "+str(x))
    log_losses.append(x)
    if abs(prev-x)<10**-5:
        break
    prev = x
    i+=1
    
    
resp = get_resp(piN)
draw_logloss_graph(log_losses, "GMM_Dataset1_RandomInit_LogLoss.png")
display_gmm(data, resp, K, "GMM_Dataset1_RandomInit")


# ### GMM Initialized with KMeans result

# In[8]:

mus, sigmas, pis = init_gmm(data, centers, preds)
pdfs = get_pdfs(mus, sigmas)
piN = get_piN(data, pdfs, pis, K)
prev = loglikelihood(piN)
print("Initial Avg. LogLoss:",str(prev))
log_losses = [prev]

i=0
while True:
    resp = get_resp(piN)
    mus, sigmas, pis = m_step(data, resp)
    pdfs = get_pdfs(mus, sigmas)
    piN = get_piN(data, pdfs, pis, K)
    x = loglikelihood(piN)
    log_losses.append(x)
    print("Iteration "+str(i+1)+" Avg. LogLoss: "+str(x))
    if abs(prev-x)<10**-5:
        break
    prev = x
    i+=1

resp = get_resp(piN)
draw_logloss_graph(log_losses, "GMM_Dataset1_KMeansInit_LogLoss.png")
display_gmm(data, resp, K, "GMM_Dataset1_KMeansInit")


# # Running for Dataset_2.txt

# In[9]:

data = read_file("Dataset_2.txt")
K = 3


# ### Kmeans

# In[10]:

r = 5
loss_array = []
preds_array = []
centers_array = []

for i in range(5):
    print("Run number",i+1)
    preds, centers, losses = kmeans(data, K)
    loss_array.append(losses[-1])
    preds_array.append(preds)
    centers_array.append(centers)
    draw_loss_graph(losses, "KMeans_Dataset2_Run"+str(i)+"_SSE.png")
    
ix = np.argmin(loss_array)
print("Lowest SSE during Run",ix,"SSE=",loss_array[ix])
display_clusters(data, centers_array[ix], preds_array[ix], "KMeans_Dataset2_Run"+str(ix))


# ### Random init GMM

# In[11]:

#160,566,754,630
# i = int(random.random()*1000)
# print(i)
# np.random.seed(i)

mus, sigmas, pis = random_init_gmm(data, K)
pdfs = get_pdfs(mus, sigmas)
piN = get_piN(data, pdfs, pis, K)
prev = loglikelihood(piN)
log_losses = [prev]
print("Initial Avg. LogLoss:",str(prev))

i=0
while True:
    resp = get_resp(piN)
    mus, sigmas, pis = m_step(data, resp)
    pdfs = get_pdfs(mus, sigmas)
    piN = get_piN(data, pdfs, pis, K)
    x = loglikelihood(piN)
    log_losses.append(x)
    print("Iteration "+str(i+1)+" Avg. LogLoss: "+str(x))
    if abs(prev-x)<10**-5:
        break
    prev = x
    i+=1
    

draw_logloss_graph(log_losses, "GMM_Dataset2_RandomInit_LogLoss.png")
resp = get_resp(piN)
display_gmm(data, resp, K, "GMM_Dataset2_RandomInit")


# ### GMM Initialized with KMeans result

# In[12]:

mus, sigmas, pis = init_gmm(data, centers, preds)
pdfs = get_pdfs(mus, sigmas)
piN = get_piN(data, pdfs, pis, K)
prev = loglikelihood(piN)
print("Initial Avg. LogLoss:",str(prev))
log_losses = [prev]

i=0
while True:
    resp = get_resp(piN)
    mus, sigmas, pis = m_step(data, resp)
    pdfs = get_pdfs(mus, sigmas)
    piN = get_piN(data, pdfs, pis, K)
    x = loglikelihood(piN)
    log_losses.append(x)
    print("Iteration "+str(i+1)+" Avg. LogLoss: "+str(x))
    if abs(prev-x)<10**-5:
        break
    prev = x
    i+=1
    
draw_logloss_graph(log_losses, "GMM_Dataset2_KMeansInit_LogLoss.png")
resp = get_resp(piN)
display_gmm(data, resp, K, "GMM_Dataset2_KMeansInit")