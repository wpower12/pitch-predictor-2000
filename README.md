## seq2seq modeling of MLB pitches. 

Todo - write an actual README.

The bulk of the actual modeling work is in the PitchModel class, found in the src/models folder. It builds a RNN with either a simple RNN cell, or an LSTM.  The data provided can either be just the pitch sequences, or it can include an additional feature vector for each at bat.  For the feature models, the feature vector is concatenated with the output of the RNN cell before passing through a single fully connected layer to produce the actual output at that point in the sequence.  

The ultimate goal is to compare the predictive accuracy of the basic models to those containing the additional features.  One would hope that the additional information could lead to more accuracte models.  

Data is collected from the MLB by way of the MLBgame python library.  To get additional information on players, the lahmanDB baseball database is also used. The scripts used to update the database, and build a full data set are all in the src/data folder.  