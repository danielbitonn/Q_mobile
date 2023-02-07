### Basic packages
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 8000)
from numpy.linalg import norm
import statistics
from scipy import stats as stat
import math

### System
import os
import time
import datetime
from datetime import datetime as dt
from datetime import timedelta
import yaml
from yaml.loader import *
import pprint
from beeprint import pp
import logging
import pickle
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', filename='/tmp/myapp.log', filemode='w')

### Files
import json
from bs4 import BeautifulSoup  
import yaml


### EDA
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use('science')
import seaborn as sns
import sweetviz as sv

### ML
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTETomek

from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree

### NLP
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('nps_chat'), nltk.download('punkt')
import openai
from openai import Embedding
import transformers
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2TokenizerFast
from typing import Set

### Updates
# from tqdm.autonotebook import tqdm as notebook_tqdm
from tqdm import *
import haversine
import folium
from collections import defaultdict
import plotly.express as px
from IPython.display import display
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import math
import geopandas as gpd
