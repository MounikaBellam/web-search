import numpy as np
import math


# loads training data
def load_train_matrix():
    # input location of train file to read
    f = open('/Mounika/SCU courses/WebSearch&InfoRetrieval/project2/train.txt', 'r')
    train_matrix = np.zeros((201, 1001), dtype=int)
    for l, line in enumerate(f):
        line_term = line.split()
        train_matrix[int(line_term[0])][int(line_term[1])] = int(line_term[2])
    print("Total lines  in train file: ", l + 1)
    f.close()
    return train_matrix


# this function gathers all the required data and do the rest of the functionality like vector formation,
# similarity calculation, top_k users, rating prediction and writes to a file
def form_vector(user_id, mov_id, ratings, matrix, available_ratings, result_5_test):
    user_v = []
    matrix_v = []
    n = len(mov_id)
    weighted_average = 0.0
    for mp in range(available_ratings, n):
        cosine_a = np.zeros(200)
        for i, line in enumerate(matrix):
            if i != 0:
                if line[mov_id[mp]] != 0:
                    for ma in range(0, available_ratings):
                        if line[mov_id[ma]] != 0 and ratings[ma] != 0:
                            user_v.append(ratings[ma])
                            matrix_v.append(line[mov_id[ma]])
                if user_v and matrix_v and len(user_v) != 1:
                    cosine_a[i - 1] = cosine_similarity(user_v, matrix_v)
                user_v = []
                matrix_v = []
        if np.any(cosine_a):
            top_k = top_k_users(cosine_a)
            weighted_average = cosine_prediction(cosine_a, top_k, matrix, mov_id[mp])
        else:
            self_sum = 0
            for ma in range(0, available_ratings):
                self_sum += ratings[ma]
            weighted_average = self_sum / available_ratings

        result_5_test.write(str(user_id))
        result_5_test.write(" ")
        result_5_test.write(str(mov_id[mp]))
        result_5_test.write(" ")
        result_5_test.write(str(int(round(weighted_average))))
        result_5_test.write("\n")
    return result_5_test

# calculates prediction
def cosine_prediction(cosine_a, top_k, matrix, movie_id):
    sum_cosine = 0.0
    weight = 0.0
    for p, line in enumerate(top_k):
        sum_cosine += cosine_a[line]
        weight += cosine_a[line] * matrix[line + 1][movie_id]
    weighted_avg = weight / sum_cosine
    return weighted_avg

# calculates top_k_users
def top_k_users(cosine_a):
    cosine_nz_len = len(np.nonzero(cosine_a)[0])
    k_value = 100
    if cosine_nz_len < k_value:
        k_values = cosine_nz_len
    else:
        k_values = k_value
    sorted_in = sorted(range(len(cosine_a)), key=lambda i: cosine_a[i])[-k_values:]
    return sorted_in


# calculates similarity
def cosine_similarity(user_v, matrix_v):
    vec_length = len(user_v)
    dot_product = 0
    user_v_sq = 0
    matrix_v_sq = 0
    for i in range(vec_length):
        dot_product += user_v[i] * matrix_v[i]
        user_v_sq += user_v[i] * user_v[i]
        matrix_v_sq += matrix_v[i] * matrix_v[i]
    cos_theta = dot_product / (math.sqrt(user_v_sq) * math.sqrt(matrix_v_sq))

    return cos_theta


# extracts test data user by user and sends to form_vector function
def test_data(userid, available_ratings):
    # input location of test file to read
    f_t = open(
        '/Mounika/SCU courses/WebSearch&InfoRetrieval/project2/test5.txt', 'r')

    user_id = userid
    cur_user_id = 0
    mov_id = []
    ratings = []
    # input location of result file to be created
    result_5_test = open('/Mounika/SCU courses/WebSearch&InfoRetrieval/project2/'
                         'result5_pred.txt', 'w+')
    for l, line in enumerate(f_t):
        line_term_test = line.split()
        cur_user_id = int(line_term_test[0])
        if cur_user_id != user_id:
            form_vector(user_id, mov_id, ratings, training_matrix, available_ratings, result_5_test)
            user_id = cur_user_id
            mov_id = []
            ratings = []

        mov_id.append(int(line_term_test[1]))
        ratings.append(int(line_term_test[2]))

    form_vector(user_id, mov_id, ratings, training_matrix, available_ratings, result_5_test)
    result_5_test.close()
    f_t.close()


training_matrix = load_train_matrix()
# starting userid of test data file, number of available ratings say (201,5), (301, 10) and (401, 20)
test_data(201, 5)
