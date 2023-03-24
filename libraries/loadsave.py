# handles loading and saving data

import pickle

###############################################################################
###############################################################################
# save & load

def save(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)


def load(filename):
    data = pickle.load(open(filename, 'rb'))
    return data
