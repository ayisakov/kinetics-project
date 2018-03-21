
# coding: utf-8

# # Kinetics Project Validation Tool
# ## This notebook can be used to validate a model against experimental data

# In[1]:


import numpy as np
import matplotlib.pyplot as pl
import pandas as pd


# ## Use the following cell to import tab-separated experiment data

# In[2]:


filename = "data/exp1420.tsv"
exp = pd.read_csv(filename, sep="\t|[ ]{1,}", engine='python', skiprows=2, names=['Time', 'A', 'D', 'U'])


# In[3]:


init = pd.read_csv(filename, sep="\t|[ ]{1,}", engine='python', skiprows=1, names=['A', 'D', 'U', 'C', 'T'], nrows=1, usecols=range(2, 7))

