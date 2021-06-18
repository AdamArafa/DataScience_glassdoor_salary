# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:10:23 2021

@author: arafa
"""

import requests 
from data_in_test import data_in

URL = 'http://127.0.0.1:5000/predict'

# defining a headrers dict for the parameters to be sent to the API

headers = {'Content-Type' : 'application/json'}
data = {'input' : data_in}

# sending get request and saving the response as response object
r = requests.get(URL, headers = headers, json = data)

r.json()
