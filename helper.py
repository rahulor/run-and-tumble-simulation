#import pandas as pd
import pickle
import bz2
def pickle_to(filename, data):
    pikd = open(filename + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()
    
def unpickle_from(filename):
    pikd = open(filename + '.pickle', 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

def compressed_pickle_to(filename, data):
    pikd = bz2.BZ2File(filename + '.pbz2', 'wb')
    pickle.dump(data, pikd)
    pikd.close()

def decompressed_pickle_from(filename):
    pikd = bz2.BZ2File(filename + '.pbz2', 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data
    
def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    pathtxt = 'data/' + fname + '.txt'
    df.to_csv(pathcsv, index=False)
    file = open(pathtxt, "w")
    text = df.to_string()
    #file.write("RESULT\n\n")
    file.write(text)
    file.close()

if __name__ == '__main__':
    pass
    