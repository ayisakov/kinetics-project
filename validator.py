
# coding: utf-8

# # Kinetics Project Validation Tool
# ## This notebook can be used to validate a model against experimental data

# In[3]:


import numpy as np
import matplotlib.pyplot as pl
import pandas as pd


# ## Use the following cell to import CSV experiment data

# In[4]:


frame = pd.read_csv("data.tsv", delimiter='\t', header=0)

