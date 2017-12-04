## Source Files

The python code and scripts used in this project are organized here. 

 * **/code** - Scripts used to gather inital data, combine it into large sets, and convert it to appropritate numpy objects. 
 * **/models** - Classes that abstract the two basic models, to simplify testing various configurations.
 * **/experiements** - Scripts that investigate parameter settints and model behaviour.
 * **/testing** - Simple scripts used during development to test other code. 

The top level scripts are used to fully train a model based on data provided. They assume that the data file pointed to by the arguments contains properly formatted numpy arrays.   

Example usage: 

````
~/src >python train_sequence_script.py ../data/full_handposbase_pitches.p feature lstm 0.01 5 50 400000 10000 test_id_str 64 32

~/src >pyhton train_window_script.py ../data/full_window_2_hbp_pitches.p 0.01 5 50 400000 10000 test_id_str 64 32
````

Both scripts contain defaults that can be used for debugging.