# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:54:00 2022

@author: n.sasidhar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from aug_model import AugNet


print(tf.__version__)


df = pd.read_excel("D:\ML_CRA\CRA_database_Scientific_Data_Publication_12102020_Compatible_ForText_02.09.2022.xlsx",sheet_name=2)

dataset = df.copy()
dataset.tail()

dataset = dataset.drop(['No.','Max Epit, mV (SCE)','ClpH', 'Min Epit, mV (SCE)','Test Solution', 'Avg MoW', 'Mosquare', 'Nsquare', 'PRE', 'Enhancers', 'Incl'], axis=1)

print(dataset.shape)
print(dataset.isna().sum())

train, val = train_test_split(dataset, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')

def df_to_dataset(dataset, shuffle=True, batch_size=32):
  dataset = dataset.copy()
  labels = dataset.pop('Avg. Epit, mV (SCE)')
  dataset[['TestMethod','ScanRatemV/s', 'HeatTreatment', 'Comments']] = dataset[['TestMethod','ScanRatemV/s', 'HeatTreatment', 'Comments']].astype('string')
    
  ds = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataset))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

def get_normalization_layer(name: string, dataset):
  # Create a Normalization layer for our feature.
  normalizer = tf.keras.layers.Normalization(axis=None, name=name+'Normalizer')

  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])
    

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

##Function for entire NLP structure. Nomenclature is confusing

def get_text_vectorization_layer(name: string, dataset):
    
    def custom_standardization(dataset):
        lowercase = tf.strings.lower(dataset)
        stripped_html = tf.strings.regex_replace(lowercase, '-1', 'na')
        return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

         
    vocab_size = 10000
    sequence_length = 200
    vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length, name=name+'TextVectorization')
    text_ds = train_ds.map(lambda x, y: x[name])
    vectorize_layer.adapt(text_ds)
    
    encoder = tf.keras.layers.Embedding(vocab_size, 64, input_length=sequence_length, name = name+'Embedding')
    embedded_input = layers.LSTM(32, name = name+'LSTM')
    return lambda feature: embedded_input(encoder(vectorize_layer(feature)))
  
    

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn', 'Cu', 'P', 'S', 'Al', 'V', 'Ta', 'Re', 'Ce', 'Ti', 'Co', 'B', 'Mg', 'Y', 'Gd', 'TestTemp', 'ClM', 'pH', 'Microstructures', 'MaterialClass']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)
  
# Categorical features encoded as string.
text_cols = ['TestMethod','ScanRatemV/s','HeatTreatment','Comments']
num_words = 10000
for header in text_cols:
  
  text_col = tf.keras.Input(shape=(1,), name=header, dtype=tf.string)
  all_inputs.append(text_col)
  
  vectorization_layer = get_text_vectorization_layer(header, train_ds)
  vectorized_text_col = vectorization_layer(text_col)
  encoded_features.append(vectorized_text_col)
  
all_features = tf.keras.layers.concatenate(encoded_features)


x = tf.keras.layers.Dense(64, activation="relu", name = 'dense1')(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
y = tf.keras.layers.Dense(64, activation="relu", name = 'dense2')(x)
y = tf.keras.layers.Dropout(0.5)(y)
z = tf.keras.layers.Dense(32, activation="relu", name = 'dense3')(y)
z = tf.keras.layers.Dropout(0.5)(z)
output = tf.keras.layers.Dense(1, name = 'dense4')(z)
model = AugNet(all_inputs, output)
model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

history = model.fit(train_ds, epochs=5000, validation_data=val_ds)


def plot_loss(history):
  plt.rc('font', size=14, weight='bold')
  plt.plot(history.history['loss'], label='Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epoch', fontweight='bold')
  plt.ylabel('Error in pitting potential (mV)', fontweight='bold')
  plt.legend()

plot_loss(history)

test_results = {}
test_results['dnn_model'] = model.evaluate(val_ds)

test_predictions = model.predict(val_ds).flatten()

a = plt.axes(aspect='equal')
plt.scatter(val['Avg. Epit, mV (SCE)'], test_predictions)
plt.xlabel('True pitting potentials (mV)', fontweight = 'bold')
plt.ylabel('Predicted pitting potentials (mV)', fontweight = 'bold')
lims = [-1600, 1600]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

metric = tfa.metrics.r_square.RSquare()
metric.update_state(val['Avg. Epit, mV (SCE)'], test_predictions)
print(metric.result())

#########################################################################################

Instance = 4
Test = pd.DataFrame(val.iloc()[Instance])
Test = Test.transpose()
Test = Test.append({'Fe': 0}, ignore_index=True)
for name, values in Test.iteritems():
    Test[name].iloc[1] = Test[name].iloc[0]

Test.to_csv('Test.csv')
Test = pd.read_csv('Test.csv', index_col=0)

test_ds = df_to_dataset(Test, shuffle=False, batch_size=2)



########################################################################################
##########Breaking NLP part for comp optimization


def get_text_vectorization_layer_sub_model(name: string, dataset, model):
    
    def custom_standardization(dataset):
        lowercase = tf.strings.lower(dataset)
        stripped_html = tf.strings.regex_replace(lowercase, '-1', 'na')
        return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

         
    vocab_size = 10000
    sequence_length = 200
    vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
    text_ds = train_ds.map(lambda x, y: x[name])
    vectorize_layer.adapt(text_ds)
    
    encoder = tf.keras.layers.Embedding(vocab_size, 64, input_length=sequence_length, weights = model.get_layer(name = name+'Embedding').get_weights())
    embedded_input = layers.LSTM(32, weights = model.get_layer(name = name+'LSTM').get_weights())
    return lambda feature: embedded_input(encoder(vectorize_layer(feature)))


all_inputs_sub = []
encoded_features_sub = []

# Numeric features.
for header in ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn', 'Cu', 'P', 'S', 'Al', 'V', 'Ta', 'Re', 'Ce', 'Ti', 'Co', 'B', 'Mg', 'Y', 'Gd', 'TestTemp', 'ClM', 'pH', 'Microstructures', 'MaterialClass']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, test_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs_sub.append(numeric_col)
  encoded_features_sub.append(encoded_numeric_col)
  
# Categorical features encoded as string.
text_cols = ['TestMethod','ScanRatemV/s','HeatTreatment','Comments']
num_words = 10000
for header in text_cols:
  
  text_col = tf.keras.Input(shape=(1,), name=header, dtype=tf.string)
  all_inputs_sub.append(text_col)
  
  vectorization_layer = get_text_vectorization_layer_sub_model(header, test_ds, model)
  vectorized_text_col = vectorization_layer(text_col)
  encoded_features_sub.append(vectorized_text_col)
  
outputs_sub = tf.keras.layers.concatenate(encoded_features_sub)

model_sub = tf.keras.Model(all_inputs_sub, outputs_sub)

##############################################################################################
    
input_sub_2 = tf.keras.Input(shape=(157,), dtype=tf.float32)

x1 = tf.keras.layers.Dense(64, activation="relu", weights = model.get_layer(name = 'dense1').get_weights())(input_sub_2)
x1 = tf.keras.layers.Dropout(0.5)(x1)
y1 = tf.keras.layers.Dense(64, activation="relu", weights = model.get_layer(name = 'dense2').get_weights())(x1)
y1 = tf.keras.layers.Dropout(0.5)(y1)
z1 = tf.keras.layers.Dense(32, activation="relu", weights = model.get_layer(name = 'dense3').get_weights())(y1)
z1 = tf.keras.layers.Dropout(0.5)(z1)
output_sub_2 = tf.keras.layers.Dense(1, weights = model.get_layer(name = 'dense4').get_weights())(z1)

model_sub_2 = AugNet(input_sub_2, output_sub_2)



############################################################################

Instance = 34
Test = pd.DataFrame(val.iloc()[Instance])
Test = Test.transpose()
Test = Test.append({'Fe': 0}, ignore_index=True)
for name, values in Test.iteritems():
    Test[name].iloc[1] = Test[name].iloc[0]

Test.to_csv('Test.csv')
Test = pd.read_csv('Test.csv', index_col=0)

test_ds = df_to_dataset(Test, shuffle=False, batch_size=2)

###############################################################################

OptHistory = []
LabelHistory = []

InitialInstance = Test
Jacobian = model_sub_2.return_jacobian([tf.Variable(model_sub.predict(test_ds))])

for t in range(2):
    sum=0
    Jacobian = model_sub_2.return_jacobian([tf.Variable(model_sub.predict(test_ds))])
    
    for i in range(24):
        if(Test.iat[0,i]>1e-5):
            Test.iat[0,i] = Test.iat[0,i] + 0.0001*Jacobian[0][0][0][0][i].numpy()
            Test.iat[1,i] = Test.iat[1,i] + 0.0001*Jacobian[0][0][0][0][i].numpy()
 
        
    OptHistory.append(Test.iloc[0,0:27])
    test_ds = df_to_dataset(Test, shuffle=False, batch_size=2)
    
    
    for i in range(1,23):
        sum = sum + Test.iat[0,i]
        
    if(sum>100):
        break

 
np.savetxt("D:\ML_CRA\SecondDNN\OptimizationResults_TextDNN.csv", OptHistory, delimiter = ",")
    
        
        