
# coding: utf-8

# In[1]:

import os
import numpy as np


# In[2]:

def read_file(file):
    with open(file, "r") as f:
        data = f.read()
        data = data.split(" \n")
    for i in range(len(data)):
        data[i] = data[i].split(" ")
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])
    data = np.array(data)
    return data


# In[3]:

data = read_file("flips.txt")
data_count = np.sum(data, axis=1, keepdims=True)
tosses = data.shape[1]


# In[4]:

def random_init():
    theta_head = np.random.uniform(size=[1,2])
    coin_pi = np.array([0.5,0.5]).reshape(1,2)
    return theta_head, coin_pi

def get_pi_p(theta_head, coin_pi, data_count, tosses):
    p = (theta_head**data_count) * ((1-theta_head)**(tosses-data_count))
    pi_p = coin_pi * p
    return pi_p

def get_resp(pi_p):
    x = np.sum(pi_p, axis=1, keepdims=True)
    resp = pi_p/x
    return resp

def get_new_pi(resp):
    s = np.sum(resp, axis=0, keepdims=True)/resp.shape[0]
    return s

def get_new_theta(resp, data_count):
    theta_head = np.sum(resp*data_count, axis=0, keepdims=True)
    s = np.sum(resp*tosses, axis=0, keepdims=True)
    return theta_head/s

def log_loss(pi_p):
    s = np.sum(pi_p, axis=1, keepdims=True)
    s = -np.sum(np.log(s))/pi_p.shape[0]
    return s


# In[5]:

np.random.seed(42)
theta_head, coin_pi = random_init()
pi_p = get_pi_p(theta_head, coin_pi, data_count, tosses)
prev = log_loss(pi_p)
print("Initial LogLoss",prev)
log_losses = [prev]
i=0
while True:
    resp = get_resp(pi_p)
    coin_pi = get_new_pi(resp)
    theta_head = get_new_theta(resp, data_count)
    pi_p = get_pi_p(theta_head, coin_pi, data_count, tosses)
    x = log_loss(pi_p)
    log_losses.append(x)
    print("Iteration "+str(i+1)+" Avg. LogLoss: "+str(x))
    if abs(prev-x)<10**-32:
        break
    prev=x
    i+=1


# In[6]:

print(theta_head)

