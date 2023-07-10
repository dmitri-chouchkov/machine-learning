import cupy as cp
import cupyx
from cupyx.scipy.sparse import linalg
from cupyx.scipy.sparse import random as srandom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
from datetime import datetime
import pandas

# produce a random matrix with the specified rows, columns, and max rank
def createRandomLRMaxRank(rows: int, columns: int, rank: int) -> (cp.ndarray, cp.ndarray):
    return (cp.random.normal(size=(rows, rank),dtype=cp.float32), cp.random.normal(size=(rank, columns), dtype=cp.float32))

# produce a random sparse mask with specified dimensions and fill percentage
def createRandomSparseMask(m: int, n: int, density: float, random_state=None, dtype=cp.float32) ->cp.sparse:
    if density < 0 or density > 1:
        raise ValueError('density expected to be 0 <= density <= 1')
    dtype = cp.dtype(dtype)
    if dtype.char not in 'fd':
        raise NotImplementedError('type %s not supported' % dtype)
    if random_state is None:
        random_state = cp.random
    elif isinstance(random_state, (int, cp.integer)):
        random_state = cp.random.RandomState(random_state)
    # use numpy instead and remove duplicates
    ind = np.random.randint(low=0,high=n*m,size=int(n*m*density),dtype=np.int64)
    ind = np.unique(ind);
    j = ind // m
    i = ind - j * m
    vals = cp.ones(ind.size) 
    return cupyx.scipy.sparse.coo_matrix(
        (vals, (cp.array(i.astype(np.int32)), cp.array(j.astype(np.int32)))), shape=(m, n), dtype=dtype).asformat('csr')

def csr_mult_dense_dense(sp: cupyx.scipy.sparse.csr_matrix, dnL: cp.ndarray, dnR: cp.ndarray) -> cupyx.scipy.sparse.csr_matrix:
    sp_m, sp_n = sp.shape
    dnL_m, dnL_n = dnL.shape
    dnR_m, dnR_n = dnR.shape
    # dnR_m, dnR_n = dnL_n, dnL_m
    # make sure that shapes match
    if sp_m != dnL_m or sp_n != dnR_n or dnL_n != dnR_m:
        raise RuntimeError("dimension of sparse matrix must match dimenson or matrix product, inner dimension of dense matrices must also match")
    if sp.dtype != cp.float32 or dnR.dtype != cp.float32 or dnL.dtype != cp.float32:
        raise RuntimeError("data must be encoded as np.float32 or (float c type)")
    if dnL.flags.c_contiguous == False or dnR.flags.c_contiguous == False:
        raise RuntimeError("dense data must be c contiguous")
    nnz = sp.nnz
    dtype = np.promote_types(sp.dtype, np.promote_types(dnL.dtype, dnR.dtype));
    data = cp.zeros(nnz, dtype=dtype)
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    # out = sp * dn
    kernel = cp.RawKernel(
            '''
            #include <cupy/complex.cuh>
            #include <cupy/carray.cuh>
            #include <cupy/atomics.cuh>
            #include <cupy/math_constants.h>
          
            __device__ inline int get_row_id(int i, int min, int max, const int *indptr) {
                int row = (min + max) / 2;
                while (min < max) {
                    if (i < indptr[row]) {
                        max = row - 1;
                    } else if (i >= indptr[row + 1]) {
                        min = row + 1;
                    } else {
                        break;
                    }
                    row = (min + max) / 2;
                }
                return row;
            }
            extern "C" __global__ void csr_mult_dense_dense(const float* SP_DATA, const int len, const int* SP_INDPTR, const int* SP_INDICES, const int SP_M, const int SP_N, 
                const float* DN_DATA_L, const float* DN_DATA_R, const int DN_K, const int* OUT_INDPTR, float* OUT_DATA) {
                #pragma unroll 1
                CUPY_FOR(i, len) {
                    int m_out = get_row_id(i, 0, SP_M - 1, &(OUT_INDPTR[0]));
                    int n_out = SP_INDICES[i];
                    OUT_DATA[i] = 0;
                    for(int j = 0; j < DN_K; j++) {
                       OUT_DATA[i] += DN_DATA_L[m_out * DN_K + j] * DN_DATA_R[j * SP_N + n_out];
                    }
                    OUT_DATA[i] = OUT_DATA[i]*SP_DATA[i];
                };
            }''',
            'csr_mult_dense_dense', jitify=False)
    kernel((sp.data.size // 1024,),(1024,),(sp.data, sp.data.size, sp.indptr, sp.indices, sp_m, sp_n,
                             dnL.data, dnR.data, dnL_n, indptr, data))
    return cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=(sp_m, sp_n))

def lr_dense_dense_diff_norm(dnL: cp.ndarray, dnR: cp.ndarray, dnL2:cp.ndarray, dnR2:cp.ndarray) -> cp.double:
    dnL_m, dnL_n = dnL.shape
    dnR_m, dnR_n = dnR.shape
    dnL2_m, dnL2_n = dnL2.shape
    dnR2_m, dnR2_n = dnR2.shape
    # dnR_m, dnR_n = dnL_n, dnL_m
    # make sure that shapes match
    if dnL_m != dnL2_m or dnL_n != dnL2_n:
        raise RuntimeError("dimensions of L should match L2")
    if dnR_m != dnR2_m or dnR_n != dnR2_n:
        raise RuntimeError("dimensions of R should match R2")
    if dnL_n != dnR_m:
        raise RuntimeError("inner dimensions must match")
    if dnL.dtype != cp.float32 or dnR.dtype != cp.float32 or dnL2.dtype != cp.float32 or dnR2.dtype != cp.float32:
        raise RuntimeError("data must be encoded as np.float32 or (float c type)")
    dtype = cp.float32
    # allocate array to hold partial sums
    data = cp.empty(dnL_m, dtype=cp.double)
    # out = sp * dn
    kernel = cp.RawKernel(
            '''
            #include <cupy/complex.cuh>
            #include <cupy/carray.cuh>
            #include <cupy/atomics.cuh>
            #include <cupy/math_constants.h>

            extern "C" __global__ void lr_dense_dense_diff_partial(const double* DN_DATA_L,
                const double* DN_DATA_R,
                const double* DN_DATA_L2,
                const double* DN_DATA_R2,
                const int DN_M,
                const int DN_K,
                const int DN_N,
                double* VEC_OUT) {
                #pragma unroll 1
                CUPY_FOR(i, DN_M) {
                    VEC_OUT[i] = 0;
                    for(int k = 0; k < DN_N; k++) {
                        // compute difference of L@R[i, k] and L2@R2[i,k]
                        double val = 0;
                        for(int j = 0; j < DN_K; j++) {
                            val += DN_DATA_L[i * DN_K + j] * DN_DATA_R[j * DN_N + k];
                            val -= DN_DATA_L2[i * DN_K + j] * DN_DATA_R2[j * DN_N + k];
                        }
                        VEC_OUT[i] += val*val;
                    }
                };
            }''',
            'lr_dense_dense_diff_partial')
    kernel((dnL_m // 1024,),(1024,),(dnL.astype(cp.double).data, dnR.astype(cp.double).data, dnL2.astype(cp.double).data, dnR2.astype(cp.double).data, dnL_m, dnL_n, dnR_n, data))
    return data.sum() ** 0.5;



# X: reduced rank left hand operand M*R matrix from previous step or random guess
# Y: reduced rank right hand operand R*N matrix from previous step or random guess
# mask: sparse {0,1} valued matrix aligned with sparse target
# Z: M*N sparse matrix with values aligned with mask. 
def ASD_iteration(X: cp.ndarray, Y: cp.ndarray, mask: cp.sparse, Z_0: cp.sparse):
    # gradients of steppest ascent for objective function with that argument held fixed
    grad_Y_f = (csr_mult_dense_dense(mask, X, Y) - Z_0) @ (Y.T) # this result is not c-contiguous
    t_x = cp.linalg.norm(grad_Y_f) ** 2 / linalg.norm(csr_mult_dense_dense(mask, cp.ascontiguousarray(grad_Y_f), Y)) ** 2 
    X_1  = X - t_x * grad_Y_f
    grad_X_f = X_1.T @ (csr_mult_dense_dense(mask, X_1, Y) - Z_0) # this result is not c-contiguous
    t_y = cp.linalg.norm(grad_X_f) ** 2 / linalg.norm(csr_mult_dense_dense(mask, X_1, cp.ascontiguousarray(grad_X_f))) ** 2
    Y_1 = Y - t_y * grad_X_f   
    return X_1, Y_1

#
def random_test(M: int, N: int, R: int, D: float, MAX_ITERATIONS: int):
    # generate a random matrix with specified max rank
    # true rank of the matrix may end up being lower but it is unlikely
    # however distribution of singular values will probably not be very uniform
    truthL,truthR = createRandomLRMaxRank(M, N, R)
    # generate a random sparse mask to act as a random sample of our low rank matrix
    mask = createRandomSparseMask(M, N, D)
    # perform a Hadamard (componentwise) multiplication to get a sparse sampling of our low rank matrix
    observed = csr_mult_dense_dense(mask, truthL, truthR) 
    
    # create a random guess for X_0 and Y_0 as our starting point 
    # in practise these should be scaled to be comparable to the data    
    X_0,Y_0 = createRandomLRMaxRank(M, N, R);
    errors = np.zeros(MAX_ITERATIONS)
    errors2 = np.zeros(MAX_ITERATIONS)
    #benchmarking
    startTime = datetime.now()
    for x in range(MAX_ITERATIONS):
        # errors[x] = cp.linalg.norm((cp.matmul(X_0, Y_0)) - truth)
        errors2[x] = linalg.norm(csr_mult_dense_dense(mask, X_0, Y_0) - observed)
        # cost is too high to compute per cycle for large data
        # errors[x] = lr_dense_dense_diff_norm(X_0, Y_0, truthL, truthR)  
        X_0, Y_0 = ASD_iteration(X_0, Y_0, mask, observed)
    
    # compute error from truth, this operation is expensive and should not be performed each iteration
    print("total error: ")
    print(lr_dense_dense_diff_norm(X_0, Y_0, truthL, truthR))
    # plot the error from observed
    plt.plot(errors2, label ="error from observed (norm)", color= (1,0,0), linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("iterations", fontsize='8')
    plt.legend(fontsize='10')
    plt.title("Rank " + str(R) +" matrix completion for " + str(M) + " by " 
              + str(N) + " with " + str(D*100) + "% known values", fontsize='10')
    print("completed in: ")
    print(datetime.now() - startTime)
    plt.show()


def save_csr(directory: str, name:str, csr_data: cupyx.scipy.sparse.csr_matrix):
    cp.save(directory + name + '_indices.npy', csr_data.indices,allow_pickle=True)
    cp.save(directory + name + '_indptr.npy', csr_data.indptr,allow_pickle=True)
    cp.save(directory + name + '_data.npy', csr_data.data,allow_pickle=True)
    cp.save(directory + name + '_shape.npy', np.asarray(csr_data.shape),allow_pickle=True)

def load_csr(directory: str, name:str):
    indices = cp.load(directory + name + '_indices.npy', allow_pickle=True)
    indptr = cp.load(directory + name + '_indptr.npy', allow_pickle=True)
    data = cp.load(directory + name + '_data.npy', allow_pickle=True)
    shape = cp.load(directory + name + '_shape.npy', allow_pickle=True)
    return cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape= (shape[0], shape[1]), dtype=data.dtype)

# execute algorithm on a cleaned up version of the movie lens database
# in particular we use movie_row in place of movieId which is not contiguous
# we also use user z-score in place of user rating to renormalize data
# this process was carried out in SQL 

def movielens_test(rating_file, data_directory, output_directory, rank = 32, MAX_ITERATIONS = 1000):
    df = pandas.read_csv(rating_file, header =0, dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int,'movie_row': int,'zscore': float})
    max_users = df['userId'].max()
    max_movies = df['movie_row'].max()
    
    # randomly split into a training and testing set
    msk = np.random.rand(len(df)) < 0.9
    training_set = df[msk]
    testing_set = df[~msk]
    
    # reindex users and movies to 0 base from 1 base, remember to reindex again when reading from data! 
    userIDs = training_set['userId'].to_numpy()-1 
    movieIDs = training_set['movie_row'].to_numpy()-1 
    ratings = training_set['zscore'].to_numpy() 
    
    # create a sparse mask using userID, movieID coordinates from training set
    csr_mask = cupyx.scipy.sparse.coo_matrix((cp.ones(userIDs.size), (cp.array(userIDs), cp.array(movieIDs))), shape=(max_users, max_movies), dtype=cp.float32).tocsr()
    
    # create a sparse matrix of observations
    csr_observed = cupyx.scipy.sparse.csr_matrix((cp.array(ratings), csr_mask.indices, csr_mask.indptr), shape= csr_mask.shape, dtype=cp.float32)
    
    # do the same thing for testing set
    userIDs_test = testing_set['userId'].to_numpy()-1 
    movieIDs_test = testing_set['movie_row'].to_numpy()-1 
    ratings_test = testing_set['zscore'].to_numpy()
    
    csr_mask_test = cupyx.scipy.sparse.coo_matrix((cp.ones(userIDs_test.size), (cp.array(userIDs_test), cp.array(movieIDs_test))), shape=(max_users, max_movies), dtype=cp.float32).tocsr()
    csr_observed_test = cupyx.scipy.sparse.csr_matrix((cp.array(ratings_test), csr_mask_test.indices, csr_mask_test.indptr), shape= csr_mask.shape, dtype=cp.float32)
    
    # save the csr matrices to disk so that we can reuse them for training 
    # we can recover the mask from observed so may as well save disk space
    save_csr(data_directory, 'csr_observed', csr_observed)
    save_csr(data_directory, 'csr_observed_test', csr_observed_test)
   
    # run algorithm, start with a random guess
    X_0,Y_0 = createRandomLRMaxRank(max_users, max_movies, rank)
    
    errors = np.zeros(MAX_ITERATIONS)
    errors2 = np.zeros(MAX_ITERATIONS)
    #benchmarking
    startTime = datetime.now()
    for x in range(MAX_ITERATIONS):
        errors[x] = linalg.norm(csr_mult_dense_dense(csr_mask_test, X_0, Y_0) - csr_observed_test)
        errors2[x] = linalg.norm(csr_mult_dense_dense(csr_mask, X_0, Y_0) - csr_observed)
        print(errors2[x])
        print(errors[x])
        X_0, Y_0 = ASD_iteration(X_0, Y_0, csr_mask, csr_observed)  
    #save results to file
    cp.save(output_directory + 'x_0.npy', X_0, allow_pickle=True)
    cp.save(output_directory + 'y_0.npy', Y_0, allow_pickle=True)
    
    #show error graph
    plt.plot(errors, label ="error from observed testing (norm)", color= (1,0,0), linewidth=0.5)
    plt.plot(errors2, label ="error from observed training (norm)", color= (0,0,0), linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("iterations", fontsize='8')
    plt.legend(fontsize='10')
    plt.title("Rank " + str(rank) +" matrix completion for 25M movie lens data", fontsize='10')
    print("completed in: ")
    print(datetime.now() - startTime)
    plt.show()    
    
def movielens_from_file(data_directory, guess_directory, output_directory, MAX_ITERATIONS, double_rank = False):
    
    # load training and testing data, use data to regenerate the masks
    csr_observed = load_csr(data_directory, 'csr_observed')
    csr_mask = cupyx.scipy.sparse.csr_matrix((cp.ones(csr_observed.data.size, dtype=csr_observed.data.dtype), 
                                              csr_observed.indices, csr_observed.indptr), shape= csr_observed.shape, 
                                             dtype=csr_observed.data.dtype)
    csr_observed_test = load_csr(data_directory, 'csr_observed_test')
    csr_mask_test = cupyx.scipy.sparse.csr_matrix((cp.ones(csr_observed_test.data.size, dtype=csr_observed_test.data.dtype), 
                                              csr_observed_test.indices, csr_observed_test.indptr), shape= csr_observed_test.shape, 
                                             dtype=csr_observed_test.data.dtype)
    
    # load guess, usually the output of a previous run   
    X_0 = cp.load(guess_directory + 'x_0.npy', allow_pickle=True)
    Y_0 = cp.load(guess_directory + 'y_0.npy', allow_pickle=True)
    
    if(double_rank):
        X_1, Y_1 = createRandomLRMaxRank(X_0.shape[0], Y_0.shape[1], X_0.shape[1])
        X_1 *= cp.linalg.norm(X_0)/cp.linalg.norm(X_1)
        Y_1 *= cp.linalg.norm(Y_0)/cp.linalg.norm(Y_1)
        X_0 = cp.column_stack((X_0/2**0.5, X_1/2**0.5))
        Y_0 = cp.row_stack((Y_0/2**0.5, Y_1/2**0.5))  
    # tracking deviation from truth
    errors = np.zeros(MAX_ITERATIONS)
    errors2 = np.zeros(MAX_ITERATIONS)
    #benchmarking
    startTime = datetime.now()
    
    # perform the algorithm
    for x in range(MAX_ITERATIONS):
        errors[x] = linalg.norm(csr_mult_dense_dense(csr_mask_test, X_0, Y_0) - csr_observed_test)
        errors2[x] = linalg.norm(csr_mult_dense_dense(csr_mask, X_0, Y_0) - csr_observed)
        print(errors2[x])
        print(errors[x])
        X_0, Y_0 = ASD_iteration(X_0, Y_0, csr_mask, csr_observed)  
    
    #save results to file
    cp.save(output_directory + 'x_0.npy', X_0, allow_pickle=True)
    cp.save(output_directory + 'y_0.npy', Y_0, allow_pickle=True)
    
    #show error
    plt.plot(errors, label ="error from observed testing (norm)", color= (1,0,0), linewidth=0.5)
    plt.plot(errors2, label ="error from observed training (norm)", color= (0,0,0), linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("iterations", fontsize='8')
    plt.legend(fontsize='10')
    plt.title("Matrix completion for 25M movie lens data", fontsize='10')
    print("completed in: ")
    print(datetime.now() - startTime)
    plt.show()     

# analyze movie lens output

def computeStratifiedError(ratings_file, output_directory, stats_directory, data_directory, allow_overshoot = False, overshoot_factor = 0.5):
    # get the original file
    df = pandas.read_csv(ratings_file, header =0, dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int,'movie_row': int, 'zscore':float})
    max_users = df['userId'].max()
    max_movies = df['movie_row'].max()    
    # create user statistics
    df_user_count = df.groupby(['userId'], as_index=True)['rating'].count().to_frame().rename(columns={'rating': 'count'})
    
    # and movie statistics
    df_movie_count = df.groupby(['movie_row'], as_index=True)['rating'].count().to_frame().rename(columns={'rating': 'count'})
         
    # load test data
    csr_observed_test = load_csr(data_directory, 'csr_observed_test')
    # recover mask from observed
    csr_mask_test = cupyx.scipy.sparse.csr_matrix((cp.ones(csr_observed_test.data.size, dtype=csr_observed_test.data.dtype), 
                                              csr_observed_test.indices, csr_observed_test.indptr), shape= csr_observed_test.shape, 
                                             dtype=csr_observed_test.data.dtype)

    # compute errors and store in host memory
    X_0 = cp.load(output_directory + 'x_0.npy', allow_pickle=True)
    Y_0 = cp.load(output_directory + 'y_0.npy', allow_pickle=True)
    
    np_data_predicted = cp.asnumpy(csr_mult_dense_dense(csr_mask_test, X_0, Y_0).data)
    np_data_true = cp.asnumpy(csr_observed_test.data)
    np_error_data = cp.asnumpy(cp.abs(csr_observed_test.data - csr_mult_dense_dense(csr_mask_test, X_0, Y_0).data))

    np_indptr = cp.asnumpy(csr_observed_test.indptr)
    np_indices = cp.asnumpy(csr_observed_test.indices)
          
    # organize errors based on total user ratings and total movie ratings
    user_breakpoints = np.asarray([32, 64, 128, 256, 512])
    movie_breakpoints = np.asarray([64, 128, 256, 512, 1024, 2048, 4096])
    
    # preallocate array to hold all these ratings, dynamically growing a list in real time is not efficient
    lengths = np.zeros((user_breakpoints.size + 1, movie_breakpoints.size + 1), dtype = np.int32)  
    error_data = np.zeros((user_breakpoints.size + 1, movie_breakpoints.size + 1, np_error_data.size // 5), dtype=np.float32)
    # this is really slow in python, better off to do it in c++
    for index in range(0,csr_observed_test.indptr.size-1):
        userID = index + 1 # users are represented as rows in the matrix output with 0 indexing instead of 1 indexing
        # get the number of total ratings for this user
        uRatings = df_user_count.loc[userID]['count']   
        # figure out which bucket we fall into based on breakpoints
        uBucket = np.searchsorted(user_breakpoints, uRatings)        
        # then for each of their ratings
        for j in range(np_indptr[index], np_indptr[index+1]):
            # get the number of total ratings for the movie
            movieID = np_indices[j] + 1
            mRatings = df_movie_count.loc[movieID]['count']
            # figure out which bucket we fall into based on breakpoints
            mBucket = np.searchsorted(movie_breakpoints, mRatings) 
            # push the error into corresponding bucket and increment array length
            error = np_error_data[j]
            if(allow_overshoot):
                if(np_data_predicted[j] > 2):
                    np_data_predicted[j] = 2
                if(np_data_predicted[j] < -2):
                    np_data_predicted[j] = -2
                error = abs(np_data_predicted[j] - np_data_true[j])
                if(np_data_true[j] >= 0):
                    if(np_data_predicted[j] > np_data_true[j]):
                        error = max(0, np_data_predicted[j] - np_data_true[j] * (1 + overshoot_factor))
                else:
                    if(np_data_predicted[j] < np_data_true[j]):
                        error = max(0, np_data_true[j] *(1 +  overshoot_factor) - np_data_predicted[j])
            error_data[uBucket][mBucket][lengths[uBucket][mBucket]] = error;
            lengths[uBucket][mBucket] += 1
    # compute average error in each bucket
    
    output_sizes = np.zeros(lengths.shape, dtype= np.int32)
    output_means = np.zeros(lengths.shape, dtype= np.float32)
    for i in range(error_data.shape[0]):
        for j in range(error_data.shape[1]):
            data = error_data[i][j][0: lengths[i][j]]
            print("[" +str(i) + ","+str(j) +"]")
            output_sizes[i][j] = data.size
            output_means[i][j] = data.mean()
    cp.save(stats_directory + 'sizes.npy', output_sizes, allow_pickle=True)
    cp.save(stats_directory + 'means.npy', output_means, allow_pickle=True)
            

def getRecomendationsForUser(userID, X_0, Y_0):
    row = cp.asnumpy(X_0[(userID-1):(userID), :] @ Y_0)
    tagged_row = np.row_stack((np.arange(1, row.size + 1, dtype=np.int32), row))
    sorted_row = tagged_row[:, (-tagged_row[1, :]).argsort()]
    return sorted_row[0].astype(np.int32), sorted_row[1]

def getSortedRatingsForUser(userID, df):
    ratings = df[df['userId'] == userID]
    ratings = ratings.sort_values('rating', ascending=False)
    return ratings['movie_row'].to_numpy(), ratings['zscore'].to_numpy()

def getMovieData(movieID, df_movies):
    try:
        row = df_movies[df_movies['movie_row'] == movieID]
        return (row['title'].values[0], row['genres'].values[0])
    except:
        return (None, None)

def analyzeOutput(rating_file, movie_file, output_directory, data_directory, userID, movie_cutoff = 1024, recs = 30):
    df = pandas.read_csv(rating_file, header =0, dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int,'movie_row': int, 'zscore':float})
    df_movies = pandas.read_csv(movie_file, header =0, dtype={'movieId': int, 'title': str, 'genres': str,'movie_row': int})
    
    # compute model
    X_0 = cp.load(output_directory + 'x_0.npy', allow_pickle=True)
    Y_0 = cp.load(output_directory + 'y_0.npy', allow_pickle=True)
    
    userRecs, scores = getRecomendationsForUser(userID, X_0, Y_0)
    userRatings, userScores = getSortedRatingsForUser(userID, df) 
    
    df_movie_count = df.groupby(['movie_row'])['rating'].count().to_frame().rename(columns={'rating': 'count'})
    df_movie_count['movie_row'] = df_movie_count.index
    
    popular_cutoff = df_movie_count[df_movie_count['count'] > movie_cutoff]
    popular_movies = np.sort(popular_cutoff['movie_row'].to_numpy());
    
    user_rating_output = []
    for i in range(min(recs, userRatings.size)):
        title, genres = getMovieData(userRatings[i], df_movies)
        user_rating_output.append((title, genres.replace("|",", "), userScores[i]))
    
    model_rating_output = []
    predictions = 0
    i = -1
    while predictions < recs:
        i += 1
        movieID = userRecs[i] 

        sort_index = np.searchsorted(popular_movies, movieID)
        if sort_index == popular_movies.size or popular_movies[sort_index] != movieID:
            continue
        
        title, genres = getMovieData(movieID, df_movies)
        if title is None:
            #prediction for invalid index this should not happen anymore after scrubbing
            print("invalid index! " + str(movieID))
            continue        
        
        # check if user has already rated this movie skip it
        if movieID in userRatings:
            continue
        predictions += 1
        
        model_rating_output.append((title, genres.replace("|",", "), scores[i]))
    return user_rating_output, model_rating_output

def showErrorHeatMaps(error_file, control_file):
    means_r64 = np.load(error_file,allow_pickle=True)
    means_r1 = np.load(control_file,allow_pickle=True)
    
    # create a heatmap for relative error:
    users = ["1-32", "33-64", "65-128", "129-256", "257-512","513+"]
    movies  = ["1-64", "65-128", "129-256", "257-512", "513-1024", "1025-2048","2049-4096", "4097+"]
    
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    
    newcolors = np.flip(np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128)))), axis = 0)
    
    print(newcolors)
    newcmp = ListedColormap(newcolors, name='OrangeBlue')    
    
    fig, ax = plt.subplots()
    
    dataset = (means_r1 - means_r64) / means_r1
    
    im = ax.imshow(dataset, cmap = newcmp, vmin = -1, vmax = 1)
    fig.colorbar(im, ax = ax)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(movies)), labels=movies)
    ax.set_yticks(np.arange(len(users)), labels=users)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")    
    # Loop over data dimensions and create text annotations.
    for i in range(len(users)):
        for j in range(len(movies)):
            text = ax.text(j, i, "{:10.2f}".format(dataset[i, j]),
                           ha="center", va="center", color="black", fontsize=8)   
    ax.set_title("normalized error with overshoot and truncating")
    plt.ylabel("ratings per user")
    plt.xlabel("ratings per movie")
    fig.tight_layout()
    plt.show()   
    
def showLogRatingDistribution(size_file):
    sizes = np.load(size_file,allow_pickle=True)
    users = ["1-32", "33-64", "65-128", "129-256", "257-512","513+"]
    movies  = ["1-64", "65-128", "129-256", "257-512", "513-1024", "1025-2048","2049-4096", "4097+"]    
    fig, ax = plt.subplots()
    
    # get a discrete color map that looks okay
    q=np.arange(0.0, 1.01, 0.05)
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in q]
    print(cmaplist)
    
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, len(q) - 1)    
    
    
    im = ax.imshow(sizes, cmap=cmap, norm=LogNorm(vmin = 100, vmax= 1000000))

    fig.colorbar(im, ax = ax)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(movies)), labels=movies)
    ax.set_yticks(np.arange(len(users)), labels=users)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor") 
    ax.set_title("number of ratings")
    plt.ylabel("ratings per user")
    plt.xlabel("ratings per movie")
    fig.tight_layout()    
    
    plt.show()

# completed in 18 minutes
# movielens_test('./ratings_clean.csv', './data_clean/', './guess_clean/', rank = 32, MAX_ITERATIONS = 2000)

# completed in 75 min, rank 64 starts overfitting
# movielens_from_file('./data_clean/','./guess_clean/','./output_clean/', MAX_ITERATIONS = 3000, double_rank = True)

# compute stratified error
# computeStratifiedError('./ratings_clean.csv', './output_clean/', './stats/', './data_clean/')

# compute stratified error with overshoot
# computeStratifiedError('./ratings_clean.csv', './output_clean/', './stats/overshoot_', './data_clean/', allow_overshoot=True, overshoot_factor=0.5)
# showErrorHeatMaps('./stats/overshoot_means.npy', './stats/rank1_means.npy')
# showLogRatingDistribution('./stats/sizes.npy')

# quick and dirty way to make a nice latex table
# TODO: escape latex inside titles
def printRatingsToLatexTable(userID, movie_cutoff, num):
    userpics, recs = analyzeOutput('./ratings_clean.csv', './movies_clean.csv','./output_clean/', './data_clean/', userID, movie_cutoff = movie_cutoff, recs = num)
    text = ""
    for i in range(num):
        text += userpics[i][0].replace("&","\&") + "&" + userpics[i][1] + "&" + recs[i][0].replace("&","\&") +"&" + recs[i][1] + " \\\\ \n"
    f = open("./"+str(userID) + "_recs.txt", "w")
    f.write(text)
    f.close()
printRatingsToLatexTable(57231, 2048, 24)




