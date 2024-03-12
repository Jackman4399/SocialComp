#Imports
import numpy as np
import csv
#Used for certain looping methods and operations, it is faster than nested loops, 
#   since it is based on C. https://docs.python.org/3/library/itertools.html
import itertools
#Simple path manipulation from python os library
import os

###  FUNCTION TO READ/LOAD CSV  ###

def load(file):
    '''
    file: The name/addresss of the file as a String
    '''
    with open(file, 'r') as f:
        data = list(csv.reader(f, delimiter=","))  
    return np.array(data)

### FUNCTION TO SAVE DATA INTO A FILE ###

def save(fname, x, delim, fmt=None):
    '''
    fname: The name of the file
    x: The array object to be saved
    delim: The delimiter separating columns
    fmt: The format
    '''
    if fmt is None:
        np.savetxt(fname, x, delimiter=delim)
    else:
        np.savetxt(fname, x, delimiter=delim, fmt=fmt)

### FUNCTION TO CREATE MATRICES FOR OUR DATA ###

def timestampMatrix(data):
    '''
    Will return a matrix of user x item on the axes and the timestamp of each rating made in the cells.
    data: The matrix of our main data with user_id, item_id, rating, and timestamp. 
    '''
    #Largest user id
    user_num= data[:, 0].astype(int).max()
    #Largest item id
    item_num = data[:, 1].astype(int).max()
    #Initiate array with zeros, int32 to save on memory
    time_matrix = np.zeros((user_num, item_num), dtype=np.int32)

    for user, item, _, timestamp in data:
        u = int(user)
        i = int(item)
        time_matrix[u-1, i-1] = timestamp
    return time_matrix

def userRatingMatrix(data):
    '''
    Will return a matrix of user x item on the axes and the rating of each user u for item i.
    data: The matrix of our main data with user_id, item_id, rating, and timestamp. 

    WARNING: The function assumes that the user_ids and item_ids are sequential. If they aren't, it will be less efficient,
    creating a matrix that leaves empty entries.
    '''
    #Largest user id
    user_num= data[:, 0].astype(int).max()
    #Largest item id
    item_num = data[:, 1].astype(int).max()
    #Initiate array with zeros, float16 to save on memory
    matrix = np.zeros((user_num, item_num), dtype=np.float16)

    for user, item, rating, _ in data:
        u = int(user)
        i = int(item)
        matrix[u-1, i-1] = rating
    return matrix

def meanCenteredMatrix(uimatrix):

    def avg_rating(u_rating):
        return u_rating[u_rating.nonzero()].mean()
    
    means = np.apply_along_axis(avg_rating, 1, uimatrix).reshape((uimatrix.shape[0], 1))
    index = np.where(uimatrix == 0)
    uimatrix = uimatrix - means
    uimatrix[index] = 0
    return uimatrix

def itemSimilarityMatrix(user_item_matrix, sim_func):
    '''
    Create and return an item similarity matrix given some data and the similarity function.
    - user_item_matrix: The matrix of size u x i, with u being the # of users and i the # of items
    - sim_func: A function that will calculate the similarity between the two users.
    '''
    #Get the number of items
    item_num = np.shape(user_item_matrix)[1]
    #Initiate with zeros again
    sim_matrix = np.zeros((item_num, item_num), dtype=np.float16)

    #Loops through all possible combinations using itertools, faster than a nested loop.
    for i1, i2 in itertools.combinations(range(item_num), 2):
        sim_value = sim_func(i1, i2, user_item_matrix)
        sim_matrix[i1, i2] = sim_value
        sim_matrix[i2, i1] = sim_value
    
    for i in range(item_num):
        sim_matrix[i, i] = np.nan
    return sim_matrix

### MAIN AND SUB FUNCTIONS FOR ADJ COSINE SIMILARITY ###

def common_users(i1_ratings, i2_ratings):
    '''
    Returns the users in common as indexes.
    - i1_ratings: The ratings of item 1
    - i2_ratings: The ratings of item 2
    '''
    line1 = np.nonzero(i1_ratings)
    line2 = np.nonzero(i2_ratings)
    return np.intersect1d(line1, line2)

def adj_cosine(i1, i2, matrix):
    '''
    Returns the weighted adjusted cosine similarity between two items
    - i1: The id of item 1. Int
    - i2: The id of item 2. Int
    - matrix: (User x Item) matrix containing ratings
    '''
    #Fetch ratings
    i1_ratings = matrix[:, i1]
    i2_ratings = matrix[:, i2]

    #Common users
    users = common_users(i1_ratings, i2_ratings)

    #Get common ratings, eliminates zeros
    i1_ratings = i1_ratings[users]
    i2_ratings = i2_ratings[users]

    if len(users) == 0:
        return 0

    user_averages = np.array([np.mean(ratings[ratings.nonzero()]) for ratings in matrix[users]])

    #Normalise
    adj_i1 = i1_ratings - user_averages
    adj_i2 = i2_ratings - user_averages

    #Dot product
    numerator = (adj_i1) @ (adj_i2)

    denominator = np.sqrt(sum((adj_i1)**2)) * np.sqrt(sum((adj_i2)**2))

    if denominator == 0:
        return np.nan

    similarity = round(numerator / denominator, 2)

    # Weighted similarity
    #The weight is the ratio of common users to total users
    weight = len(users) / matrix.shape[0]  
    weighted_similarity = similarity * weight

    return weighted_similarity

### NEIGHBOURHOOD FUNCTIONS ###

def k_neighbourhood(neigh, k):
    '''
    Returns the k nearest neighbours of the item. An array of indices.
    - neigh: The similarity column of the item in the item similarity matrix.
    - k: number of neighbours to fetch.
    '''
    #Count nan instances
    num_nans = np.sum(np.isnan(neigh))
    #Sort array, remove all zero/null entries
    sorted = np.flip(np.argsort(neigh[neigh.nonzero()]))
    #Remove all nans, sorting brings all nan entries to the front
    filtered = sorted[num_nans:]

    #Error checking to avoid index out of bounds. k = -1 for FULL neighbourhood
    if k > len(filtered) or k == -1:
        k = len(filtered)

    return filtered[:k]

### HELPER FUNCTIONS AND PREDICTION FUNCTIONS ###

def calc_time_weights(time_matrix):
    '''
    Use a vectorized operation to return a normalised timestamp matrix.
    - time_matrix: The user x item timestamp matrix. Each cell contains the timestamp from the training data.
    '''
    # Find the minimum and maximum timestamps
    min_timestamp = np.min(time_matrix)
    max_timestamp = np.max(time_matrix)

    # Define a function to normalize a timestamp
    def normalise_timestamp(timestamp):
        return (timestamp - min_timestamp) / (max_timestamp - min_timestamp)

    normalise = np.vectorize(normalise_timestamp)
    return normalise(time_matrix)

def round_to_half(number):
    '''
    Rounds the input to an accepted value of 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
    - number: Any number, expected to be a decimal.
    '''
    rounded = round(number * 2) / 2
    return min(max(rounded, 1.0), 5.0)

def item_based_pred(user, item, matrix, sim_matrix, time_weights, k = 1000):
    '''
    Returns the prediction for the user's rating of the item.
    - user: id of user. Int
    - item: id of item. Int
    - matrix: The user x item matrix containing all ratings.
    - sim_matrix: The item similarity matrix. 
    '''

    #Value to return when calculation error is encountered
    ratings =  matrix[:, item]
    defaultValue = np.mean(ratings)
    if np.isnan(defaultValue) or len(ratings) == 0:
        defaultValue = 2.5
    else:
        defaultValue = round_to_half(defaultValue)

    items = k_neighbourhood(sim_matrix[item], k)

    if len(items) == 0:
        return defaultValue

    common = np.intersect1d(matrix[user].nonzero(), items)

    user_ratings = matrix[user, common]
    item_sim = sim_matrix[common, item]

    weights = time_weights[user, common]

    #Apply weights
    item_sim = item_sim * weights

    #Gets positive similarities only.
    pos = np.where(item_sim > 0)

    # Count the number of non-zero ratings for each item
    counts = np.count_nonzero(matrix, axis=0)

    # Find the indices of items that have at least 5 ratings
    least_5_ratings = np.where(counts >= 5)

    pos = np.intersect1d(pos, least_5_ratings)

    numerator = (item_sim[pos]) @ user_ratings[pos]

    denominator = sum(np.abs(item_sim[pos]))

    if denominator == 0:
        return defaultValue

    output = numerator/denominator

    return round_to_half(output)

### PREDICT AND OUTPUT ###

def predict(train, pred, out_path):
    '''
    Calculated the predictions for a set of data without ratings.
    - train: The path to the data set used for training/with ratings.
    - pred: The test data without any
    '''
    # Load the data
    data = load(train)
    to_pred = load(pred)

    #Generate user matrix
    matrix = userRatingMatrix(data)

    c_matrix = meanCenteredMatrix(matrix)

    #Where to save the similarity matrix. Will take longer if file does not exist.
    file = "centered_sim.txt"

    if os.path.isfile(file):
        item_sim_matrix = np.loadtxt(file, dtype = np.float16, delimiter=",")
    else:
        #item_sim_matrix = itemSimilarityMatrix(matrix, adj_cosine)
        item_sim_matrix = itemSimilarityMatrix(c_matrix, adj_cosine)
        save(file, item_sim_matrix, ",")

    #Take user and item colum from the test set without ratings
    predict = to_pred[:, :2].astype(np.int32)
    timestamps = to_pred[:, 2].astype(np.int32)

    time_matrix = timestampMatrix(data)

    timeweights = calc_time_weights(time_matrix)

    predictions = []

    neigh_num = -1

    for u, i in predict:
        val = item_based_pred(u-1, i-1, matrix, item_sim_matrix, timeweights, neigh_num)
        predictions.append(val)

    #Stack and save in a csv file
    output = np.column_stack((predict, np.array(predictions), timestamps))
    save(out_path, output, delim=',', fmt=['%d', '%d', '%.1f', '%d'])


### ERROR MEASUREMENT FUNCTIONS ###

def mae(pred, true):
    '''
    Mean Absolute Error for the predictions and the ground truth.
    - pred: the prediction values (by the model)
    - real: the actual values (ground truth)
    '''
    return np.mean(np.abs(pred - true))

predict("train_100k_withratings.csv", "test_100k_withoutratings.csv", "results.csv")