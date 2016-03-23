import numpy as np
import theano
import math
import csv
import keras
import sklearn
import random
import scipy
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error , mean_absolute_error
from geopy import distance

def geodistance( coords1 , coords2 ):
  lat1 , lon1 = coords1[ : 2]
  lat2 , lon2 = coords2[ : 2]
  try: return distance.vincenty( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
  except: return distance.great_circle( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0

def geoloss( a , b ): 
  aa = theano.tensor.deg2rad( a )
  bb = theano.tensor.deg2rad( b )
  sin_lat1 = theano.tensor.sin( aa[:,0] )
  cos_lat1 = theano.tensor.cos( aa[:,0] )
  sin_lat2 = theano.tensor.sin( bb[:,0] )
  cos_lat2 = theano.tensor.cos( bb[:,0] )
  delta_lng = bb[:,1] - aa[:,1]
  cos_delta_lng = theano.tensor.cos(delta_lng)
  sin_delta_lng = theano.tensor.sin(delta_lng)
  d = theano.tensor.arctan2(theano.tensor.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
  return theano.tensor.mean( 6371.0088 * d , axis = -1 )

percent = 0.8
hidden_dim = 400

print ("Reading text data for regression and building representations...")
data1 = [ ( [ float(row[i]) for i in range(len(row) - 2) ] , ( float( row[ len(row) - 2 ] ) , float( row[ len(row) - 1 ] ) ) ) for row in csv.reader( open("default_features_1059_tracks.txt"), delimiter=',', quoting=csv.QUOTE_NONE) ]
np.random.seed(0)
random.shuffle( data1 )
train_size1 = int(len(data1) * percent)
train_matrix1 = np.array( [ features for ( features, label ) in data1[0:train_size1] ] )
test_matrix1 = np.array( [ features for ( features, label ) in data1[train_size1:-1] ] )
train_labels1 = [ label for ( features , label ) in data1[0:train_size1] ]
test_labels1 = [ label for ( features , label ) in data1[train_size1:-1] ]
train_matrix1 = preprocessing.scale( train_matrix1 )
test_matrix1 = preprocessing.scale( test_matrix1 )
data2 = [ ( [ float(row[i]) for i in range(len(row) - 2) ] , ( float( row[ len(row) - 2 ] ) , float( row[ len(row) - 1 ] ) ) ) for row in csv.reader( open("default_plus_chromatic_features_1059_tracks.txt"), delimiter=',', quoting=csv.QUOTE_NONE) ]
np.random.seed(0)
random.shuffle( data2 )
train_size2 = int(len(data2) * percent)
train_matrix2 = np.array( [ features for ( features, label ) in data2[0:train_size2] ] )
test_matrix2 = np.array( [ features for ( features, label ) in data2[train_size2:-1] ] )
train_labels2 = [ label for ( features , label ) in data2[0:train_size2] ]
test_labels2 = [ label for ( features , label ) in data2[train_size2:-1] ]
train_matrix2 = preprocessing.scale( train_matrix2 )
test_matrix2 = preprocessing.scale( test_matrix2 )

print ("")
np.random.seed(0)
print ("Method = Linear ridge regression - Default features")
model = KernelRidge( kernel='linear' )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Linear ridge regression - Default features + chromatic features")
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = Random forest regression - Default features")
model = RandomForestRegressor( n_estimators=25 )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Random forest regression - Default features + chromatic features")
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = Kernel ridge regression with RBF kernel - Default features")
model = KernelRidge( kernel='rbf' )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Kernel ridge regression with RBF kernel - Default features + chromatic features")
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = Kernel ridge regression with polynomial kernel - Default features")
model = KernelRidge( kernel='poly' , degree=2 )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Kernel ridge regression with polynomial kernel - Default features + chromatic features")
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = MLP - Default features")
train_labels_aux = np.array( [ round( ( lon + 180 ) / 2 ) + ( 180 * round( ( lat + 90 ) / 2 ) )  for (lat,lon) in train_labels1 ] )
train_labels_aux = np.array( preprocessing.LabelEncoder( ).fit_transform( train_labels_aux ) )
num_classes = len( set( train_labels_aux ) )
train_labels_aux = np_utils.to_categorical(train_labels_aux , num_classes )
model2 = Sequential( )
model2.add(Dense(hidden_dim, input_dim=train_matrix1.shape[1], init='uniform', activation='sigmoid'))
model2.add(Dropout(0.1))
model2.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model2.add(Dropout(0.1))
model2.add(Dense(hidden_dim / 4, activation='sigmoid', init='uniform'))
model2.add(Dropout(0.1))
model2.add(Dense(hidden_dim / 8, activation='sigmoid', init='uniform'))
model2.add(Dropout(0.1))
model2.add(Dense( num_classes , activation='softmax' , init='uniform' ))
model2.compile(loss='categorical_crossentropy', optimizer='adam')
model2.fit( train_matrix1 , train_labels_aux , nb_epoch=1000, batch_size=16, verbose=1)
model = Sequential()
model.add(Dense(hidden_dim, input_dim=train_matrix1.shape[1], init='uniform', activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model.add(Dropout(0.1))
model.add(Dense(hidden_dim / 4, activation='sigmoid', init='uniform'))
model.add(Dropout(0.1))
model.add(Dense(hidden_dim / 8, activation='sigmoid', init='uniform'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss=geoloss, optimizer='adam')
for i in range( len( model.layers ) - 1 ) : model.layers[i].set_weights( model2.layers[i].get_weights() )
model.fit( train_matrix1 , train_labels1 , nb_epoch=1000, batch_size=16 , verbose=1)
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )


print ("Method = MLP - Default features + chromatic features") 
#np.random.seed(0)
#model = Sequential()
#model.add(Dense(hidden_dim, input_dim=train_matrix2.shape[1], init='glorot_uniform', activation='sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(hidden_dim / 2, activation='sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(hidden_dim / 4, activation='sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(hidden_dim / 8, activation='sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(2, activation='sigmoid'))
#model.compile(loss=geoloss, optimizer='adam')
#model.fit( train_matrix2 , train_labels2 , nb_epoch=1000, batch_size=8)
#results = model.predict( test_matrix2 )
#print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
#print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
