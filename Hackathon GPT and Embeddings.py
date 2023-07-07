# Databricks notebook source
# MAGIC %sh pip install openai-whisper 
# MAGIC     openai 
# MAGIC     num2words 
# MAGIC     matplotlib 
# MAGIC     plotly 
# MAGIC     scipy 
# MAGIC     scikit-learn 
# MAGIC     pandas 
# MAGIC     tiktoken 
# MAGIC     cosine_similarity 
# MAGIC     numpy 
# MAGIC     pandas 
# MAGIC     pysparkrequests 
# MAGIC     azure-cognitiveservices-speech 
# MAGIC     ffmpeg-python 
# MAGIC     whisper 
# MAGIC     Js2Py

# COMMAND ----------

#Import Libraries
import os
import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
from pyspark.sql import SparkSession
import requests
import json
import azure.cognitiveservices.speech as speechsdk
from IPython.display import HTML, Audio, display
from base64 import b64decode
import scipy
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import whisper
import platform
from js2py import eval_js
from pyspark.sql import SparkSession
from IPython.display import display

# COMMAND ----------

### Open AI Authorization and Keys
openai.api_type = "azure"
openai.api_version = "2023-05-15" #"2022-12-01"#
openai.api_base = os.getenv("https://dev-da2i-openai.openai.azure.com/")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("acf91e9be7c04d0894bd1ea837b94e97")
API_KEY = "acf91e9be7c04d0894bd1ea837b94e97" 
RESOURCE_ENDPOINT = "https://dev-da2i-openai.openai.azure.com/" 
url = "https://dev-da2i-openai.openai.azure.com/openai/deployments?api-version=2022-12-01" 
r = requests.get(url, headers={"api-key": API_KEY})
openai.api_key = "acf91e9be7c04d0894bd1ea837b94e97"
openai.api_base = "https://dev-da2i-openai.openai.azure.com/" 

print(f"I think this is the error: {r.text}")

# COMMAND ----------
l_path = os.getcwd()
print(l_path)
iconVenue = os.path.join(l_path, 'icon_venue_locations.csv')
print(iconVenue)
# Read the CSV file
df = pd.read_csv(iconVenue)

# Specify the columns to keep
columns_to_keep = ['venue','venue_name', 'id', 'deck', 'location', ]

# Filter the DataFrame to keep only the specified columns
filtered_df = df[columns_to_keep]

# Set the display width to None for unlimited width
pd.set_option('display.width', None)

# Display the filtered DataFrame using Databricks display function
display(filtered_df)

# COMMAND ----------

# Create a dictionary mapping old locations to new locations
location_mapping = {
    'Aft': 'back',
    'Forward': 'front',
    'Midship': 'middle',
    'Midship Port': 'middle port side',
    'Midship Starboard': 'middle starboard side'
}

# Apply the mapping to create the new column
filtered_df['locations_mod'] = filtered_df['location'].map(location_mapping)

# Print the modified DataFrame
display(filtered_df)

# COMMAND ----------

# Create the sentence

# Desired output: The [venue_name] is located on deck [deck] near the [locations_mod] of the ship
filtered_df['sentence'] = "The " + filtered_df['venue_name'].fillna('') + " is located on deck " + filtered_df['deck'].astype(str).fillna('') + " near the " + filtered_df['locations_mod'].fillna('') + " of the ship"

with pd.option_context('display.max_colwidth', None):
    # display(filtered_df['sentence'])
    display(filtered_df)

# COMMAND ----------

# Perform some light data cleaning by removing redundant whitespace and cleaning up the punctuation to prepare the data for tokenization.

pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

filtered_df['sentence']= filtered_df["sentence"].apply(lambda x : normalize_text(x))

# COMMAND ----------

tokenizer = tiktoken.get_encoding("cl100k_base")
filtered_df['n_tokens'] = filtered_df["sentence"].apply(lambda x: len(tokenizer.encode(x)))
filtered_df = filtered_df[filtered_df.n_tokens<8192]
len(filtered_df)

# COMMAND ----------

filtered_df

# COMMAND ----------

# Understand the n_tokens column a little more as well how text ultimately is tokenized
sample_encode = tokenizer.encode(filtered_df.sentence[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)
decode

# COMMAND ----------

# If you check the length of the decode variable, you'll find it matches the first number in the n_tokens column.
len(decode)

# COMMAND ----------

#  Pass the documents to the embeddings model, it will break the documents into tokens similar (though not necessarily identical) to the examples above and then convert the tokens to a series of floating point numbers that will be accessible via vector search. These embeddings can be stored locally or in an Azure Database. As a result, each bill will have its own corresponding embedding vector in the new ada_v2 column on the right side of the DataFrame.

filtered_df['ada_v2'] = filtered_df["sentence"].apply(lambda x : get_embedding(x, engine = 'text-embedding-ada-002')) # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

# COMMAND ----------

# print(filtered_df['ada_v2'])
filtered_df

# COMMAND ----------
r_path = 'C:\\Users\\dguti\\OneDrive - Royal Caribbean Group\\Repos\\d-id_streams\\recordings'
# Load Voice Recording
best_ship_path = f"{r_path}/what_is_the_best_ship_for_a_family.m4a"
breakfast_path = f"{r_path}/where_can_I_have_breakfast.m4a"
pizza_path = f"{r_path}/where_can_I_have_pizza.m4a"
main_dining_path = f"{r_path}/where_is_the_main_dining_room.m4a"

model = whisper.load_model("base")
result = model.transcribe(pizza_path)

user_input = result['text']

# Define Embeddings Function and GPT Function
def search_docs(df, user_query, top_n=3, threshold=0.3, to_print=True):
    embedding = get_embedding(
        user_query,
        engine="text-embedding-ada-002" # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df[df["similarities"] >= threshold]
        .sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res

res_embedding = search_docs(filtered_df, user_input, top_n=1)
#res_embedding = res_embedding.to_json()

if float(res_embedding['similarities']) > 0.8:
    text3 = res_embedding['sentence'].to_json().replace('"', '').replace(':', '').replace('{','').replace('}','')[2:]
else:
    input_prompt = "Assume the role of a virtual assistant for Royal Caribbean Group designed to help guests have the best cruise vacation onboard a Royal caribbean ship. Answer questions succinctly and in a friendly manner. Do not offer products or cruises from competitors. Make fun of competitors when applicable. Only provide a response for what you are asked. Do not elaborate beyond what you've been asked. Only state facts. Don't mention dates or times. Don't display code. Based on all of these assumptions,  "
    res_gpt = openai.Completion.create(
        #engine="gpt-4",#"gpt-35-turbo", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        #prompt="Assistant is a virtual assistant for Royal Caribbean Cruise guests. Q: "+user_input ,
        deployment_id="gpt-35-turbo",
        prompt=input_prompt + user_input + "/n",
        temperature = 0.01, #How creative the response is
        frequency_penalty=0, #How often things are repeated
        top_p=0.95,
        presence_penalty=0,
        stop="/n",
        max_tokens=256,
    )
    text3 = res_gpt['choices'][0]['text'].replace('\n', '').replace(' .', '.').replace('<|im_end|>', '').replace('# Answer', '').replace('# Prompt', '').strip()
    text3 = text3.split(input_prompt)[0]

print(text3)