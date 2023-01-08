import streamlit as st
import os
import json
import pandas as pd
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import json
import torch
from sentence_transformers import util

def cosine_sim(em1, em2):
    return util.cosine_sim(em1, em2)

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

embedding = TransformerDocumentEmbeddings('xlm-roberta-base')

embeddings = []

with open('mydata.json','r+') as f:
    embeddings = json.load(f)

for k in embeddings:
    k['embedding'] = torch.Tensor(k['embedding'])

def N_max_elements(list, N):
            result_list = []
        
            for i in range(0, N): 
                maximum = 0
                maxpos = 0
                for j in range(len(list)):     
                    if list[j] > maximum:
                        maximum = list[j]
                        maxpos = j
                        
                list.remove(maximum)
                result_list.append(maxpos)
                
            return result_list


st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

inp_text = st.text_input('Enter some text','')

if(inp_text != ''):

    input_text = inp_text.replace(",","")

    inp = Sentence(input_text)
    embedding.embed(inp)

    inp_embedding = inp.get_embedding()

    sim = []

    for i in range (0, 100):
        sim.append(util.cos_sim(inp_embedding, embeddings[i]['embedding'])[0])


    result_list = N_max_elements(sim, 10)

    res_ids = []
    res_contents = []
    res_file_names = []
    for res in result_list:
        res_ids.append(res+1)
    
    result_df = pd.DataFrame({'id': res_ids})

    with open('res_data.json', 'w') as f:
        f.write(result_df.to_json(orient='records'))


    for k in res_ids:
        st.write("Result page", k, ": https://github.com/susindhar21/precedents/documents/"+str(k)+".jpg\n")


