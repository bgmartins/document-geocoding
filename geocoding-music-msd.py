import healpy as H
import numpy as np
import fnmatch
import os
import theano
import math
import csv
import keras
import sklearn
import random
import scipy
import hdf5_getters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

def latlon2healpix( lat , lon , res ):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    xs = ( math.cos(lat) * math.cos(lon) )
    ys = ( math.cos(lat) * math.sin(lon) )
    zs = ( math.sin(lat) )
    return H.vec2pix( int(res) , xs , ys , zs )

def healpix2latlon( code , res ):
    [xs, ys, zs] = H.pix2vec( int(res) , code )
    lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
    lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
    return [ lat , lon ]

percent = 0.8
hidden_dim = 1000
resolution = 1024
max_sequence_len = 1000
max_epochs = 1500

print ("Reading text data for regression and building representations...")
coordinates_artist = dict( )
for row in open("MillionSongSubset/AdditionalFiles/subset_artist_location.txt").readlines():
  row = row.split("<SEP>")
  coordinates_artist[row[0]] = ( float( row[1] ) , float( row[2] ) )
data1 = [ ]
data2 = [ ]
data3 = [ ]
for root, dirnames, filenames in os.walk('MillionSongSubset/data'):
  for datafile in fnmatch.filter(filenames, '*.h5'):
    datafile = os.path.join( root, datafile )
    datafile = hdf5_getters.open_h5_file_read( datafile )
    artist = hdf5_getters.get_artist_id(datafile)
    if coordinates_artist.has_key( artist ) :
      timbre = hdf5_getters.get_segments_timbre(datafile)
      pitch = hdf5_getters.get_segments_pitches(datafile)
      data1.append( ( np.average( timbre , axis=0 ), ( coordinates_artist[ artist ][0], coordinates_artist[ artist ][1] ) ) )
      data2.append( ( np.hstack( ( np.average( timbre , axis=0 ) , np.average( pitch , axis=0 ) ) ) , ( coordinates_artist[ artist ][0], coordinates_artist[ artist ][1] ) ) )
      combined = [ np.hstack( ( timbre[i] , pitch[i] ) ) for i in range( timbre.shape[0] ) ]
      data3.append( ( combined , ( coordinates_artist[ artist ][0], coordinates_artist[ artist ][1] ) ) ) 
    datafile.close( )
np.random.seed(0)
np.random.shuffle( data1 )
train_size1 = int(len(data1) * percent)
train_matrix1 = np.array( [ features for ( features, label ) in data1[0:train_size1] ] )
test_matrix1 = np.array( [ features for ( features, label ) in data1[train_size1:-1] ] )
train_labels1 = [ label for ( features , label ) in data1[0:train_size1] ]
test_labels1 = [ label for ( features , label ) in data1[train_size1:-1] ]
train_matrix1 = preprocessing.scale( train_matrix1 )
test_matrix1 = preprocessing.scale( test_matrix1 )
np.random.seed(0)
np.random.shuffle( data2 )
train_size2 = int(len(data2) * percent)
train_matrix2 = np.array( [ features for ( features, label ) in data2[0:train_size2] ] )
test_matrix2 = np.array( [ features for ( features, label ) in data2[train_size2:-1] ] )
train_labels2 = [ label for ( features , label ) in data2[0:train_size2] ]
test_labels2 = [ label for ( features , label ) in data2[train_size2:-1] ]
train_matrix2 = preprocessing.scale( train_matrix2 )
test_matrix2 = preprocessing.scale( test_matrix2 )
np.random.seed(0)
np.random.shuffle( data3 )
train_size3 = int(len(data3) * percent)
train_matrix3 = [ features for ( features, label ) in data3[0:train_size3] ]
test_matrix3 = [ features for ( features, label ) in data3[train_size3:-1] ]
max = np.max( [ len(train_matrix3) for i in train_matrix3 + test_matrix3 ] )
print max
for aux in train_matrix3: 
  while len(aux) < max: aux.append( np.zeros( train_matrix3[0][0].shape ) )
for aux in test_matrix3:
  while len(aux) < max: aux.append( np.zeros( train_matrix3[0][0].shape ) )
max = np.mean( [ len(train_matrix3) for i in train_matrix3 + test_matrix3 ] )
print max
train_labels3 = [ label for ( features , label ) in data3[0:train_size3] ]
test_labels3 = [ label for ( features , label ) in data3[train_size3:-1] ]

print ("Method = Stack of LSTMs - Default features + chromatic features")
encoder = preprocessing.LabelEncoder( )
train_labels_aux = np.array( [ latlon2healpix( lat , lon , resolution ) for (lat,lon) in train_labels3 ] )
train_labels_aux = np.array( encoder.fit_transform( train_labels_aux ) )
num_classes = len( set( train_labels_aux ) )
train_labels_aux = np_utils.to_categorical(train_labels_aux , num_classes )
model = Sequential( )
model.add(LSTM(output_dim=hidden_dim, input_length=max_sequence_len , activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=hidden_dim / 2 , activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(hidden_dim / 2, init='uniform', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model.add(Dropout(0.25))
model.add(Dense( num_classes , activation='softmax' , init='uniform' )) 
model.compile(loss='categorical_crossentropy', optimizer='adam') 
model.fit( train_matrix3 , train_labels_aux , nb_epoch=max_epochs, batch_size=16, verbose=1)
results = encoder.inverse_transform( np_utils.categorical_probas_to_classes( model.predict( test_matrix3 ) ) )
results = np.array( [ healpix2latlon( code , resolution ) for code in results ] )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels3[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels3[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = Linear ridge regression - Default features")
model = KernelRidge( kernel='linear' )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Linear ridge regression - Default features + chromatic features")
model = KernelRidge( kernel='linear' )
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = Random forest regression - Default features")
model = RandomForestRegressor( n_estimators=100 , random_state=0 )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Random forest regression - Default features + chromatic features")
model = RandomForestRegressor( n_estimators=100 , random_state=0 )
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = Kernel ridge regression with RBF kernel - Default features")
model = KernelRidge( kernel='rbf' )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Kernel ridge regression with RBF kernel - Default features + chromatic features")
model = KernelRidge( kernel='rbf' ) 
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = Kernel ridge regression with polynomial kernel - Default features")
model = KernelRidge( kernel='poly' , degree=2 )
model.fit( train_matrix1 , train_labels1 )
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Kernel ridge regression with polynomial kernel - Default features + chromatic features")
model = KernelRidge( kernel='poly' , degree=2 )
model.fit( train_matrix2 , train_labels2 )
results = model.predict( test_matrix2 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = Random forest classification - Default features")
encoder = preprocessing.LabelEncoder( )
train_labels_aux = np.array( [ latlon2healpix( lat , lon , resolution ) for (lat,lon) in train_labels1 ] )
train_labels_aux = np.array( encoder.fit_transform( train_labels_aux ) )
model = RandomForestClassifier( n_estimators=100 , random_state=0 )
model.fit( train_matrix1 , train_labels_aux )
results = encoder.inverse_transform( model.predict( test_matrix1 ) )
results = np.array( [ healpix2latlon( code , resolution ) for code in results ] )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = Random forest classification - Default features + chromatic features")
encoder = preprocessing.LabelEncoder( )  
train_labels_aux = np.array( [ latlon2healpix( lat , lon , resolution ) for (lat,lon) in train_labels2 ] )                                                                                    
train_labels_aux = np.array( encoder.fit_transform( train_labels_aux ) ) 
model = RandomForestClassifier( n_estimators=100 , random_state=0 )
model.fit( train_matrix2 , train_labels_aux )
results = encoder.inverse_transform( model.predict( test_matrix2 ) )
results = np.array( [ healpix2latlon( code , resolution ) for code in results ] )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
np.random.seed(0)
print ("Method = MLP classification - Default features")
encoder = preprocessing.LabelEncoder( )
train_labels_aux = np.array( [ latlon2healpix( lat , lon , resolution ) for (lat,lon) in train_labels1 ] )
train_labels_aux = np.array( encoder.fit_transform( train_labels_aux ) )
num_classes = len( set( train_labels_aux ) )
train_labels_aux = np_utils.to_categorical( train_labels_aux , num_classes )
model = Sequential( )
model.add(Dense(hidden_dim, input_dim=train_matrix1.shape[1], init='uniform', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model.add(Dropout(0.25))
model.add(Dense( num_classes , activation='softmax' , init='uniform' ))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit( train_matrix1 , train_labels_aux , nb_epoch=max_epochs, batch_size=16, verbose=1)
results = encoder.inverse_transform( np_utils.categorical_probas_to_classes( model.predict( test_matrix1 ) ) )
results = np.array( [ healpix2latlon( code , resolution ) for code in results ] )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = MLP classification - Default features + chromatic features")
np.random.seed(0)
encoder = preprocessing.LabelEncoder( )
train_labels_aux = np.array( [ latlon2healpix( lat , lon , resolution ) for (lat,lon) in train_labels2 ] )
train_labels_aux = np.array( encoder.fit_transform( train_labels_aux ) )
num_classes = len( set( train_labels_aux ) )                                                              
train_labels_aux = np_utils.to_categorical(train_labels_aux , num_classes )
model = Sequential( ) 
model.add(Dense(hidden_dim, input_dim=train_matrix2.shape[1], init='uniform', activation='sigmoid')) 
model.add(Dropout(0.25)) 
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform')) 
model.add(Dropout(0.25))
model.add(Dense( num_classes , activation='softmax' , init='uniform' )) 
model.compile(loss='categorical_crossentropy', optimizer='adam') 
model.fit( train_matrix2 , train_labels_aux , nb_epoch=max_epochs, batch_size=16, verbose=1) 
results = encoder.inverse_transform( np_utils.categorical_probas_to_classes( model.predict( test_matrix2 ) ) )
results = np.array( [ healpix2latlon( code , resolution ) for code in results ] )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )

print ("Method = MLP regression - Default features") 
np.random.seed(0)
model = Sequential()
model.add(Dense(hidden_dim, input_dim=train_matrix1.shape[1], init='uniform', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss=geoloss, optimizer='adam')
model.fit( train_matrix1 , train_labels1 , nb_epoch=max_epochs, batch_size=16, verbose=1)
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels1[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Method = MLP regression - Default features + chromatic features")
np.random.seed(0)
model = Sequential()
model.add(Dense(hidden_dim, input_dim=train_matrix2.shape[1], init='uniform', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(hidden_dim / 2, activation='sigmoid', init='uniform'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss=geoloss, optimizer='adam')
model.fit( train_matrix2 , train_labels2 , nb_epoch=max_epochs, batch_size=16, verbose=1)
results = model.predict( test_matrix1 )
print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels2[i] ) for i in range(results.shape[0]) ] ) ) )
