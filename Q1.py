import numpy as np
from Utils.preprocessing import *
import os
import glob
from tqdm import tqdm
import pickle as pkl

data_loc = os.path.join(os.getcwd(), "Data/Humor,Hist,Media,Food/")
data = []
error_files = []
for filename in glob.glob(data_loc + "*"):
    with open(filename, 'r', encoding='latin-1') as f:
        name = str(filename).split("/")[-1]
        content = f.read()
        data.append({'file':name,'content':content})

# filters the data
for files in tqdm(data):
    files['filtered_content'] = filter(files['content'])

for file in data:
    print(f"Text for file- {file['content']}")
    print(file['content'][:20])
    print(f"Cleaned text for file- {file['filtered_content']}")
    print(file['filtered_content'][:20])
    break

data.sort(key=lambda x: x['file'])

pkl.dump(data,open('Data/docs.pkl','wb'))