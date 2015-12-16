import os
import sys
import numpy as np
from numpy import linalg as nplinalg

from scipy import linalg

import time
import pyspark

MIN_EPS = 0.005
TINY_EPS = 0.001 # convergence condition


def calculate_loss(pred_matrix, true_matrix):
    rmse = (np.abs(pred_matrix-true_matrix) ** 2).sum() ** .5
    percent = rmse/np.abs(true_matrix).sum()
    print('RMSE: %f, percent:%f' % (rmse,percent))
    return rmse,percent


def get_worker_id_for_position(idx):
    #  determine the worker id from strata_size
    strata_size = strata_size_bc.value
    block_start = np.floor(idx / strata_size)*strata_size
    return block_start, block_start + strata_size


def blockify_matrix(worker_id, partition):
    blocks = set()
    for v in partition:
        blocks.add(v)

    for item in blocks:
        yield item


def filter_block_for_iteration(num_iteration, block_index):
    return block_index == mod_workers(block_index + mod_workers(num_iteration))

def mod_workers(n):
    return np.mod(n,num_workers)

def lasso(X, y, lbd, miniter = 10, maxiter = 5000, tol = 1e-5, t = 0.001):
    # initialize beta
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    _,p = X.shape
    beta = np.zeros([p,1]).astype(np.float64)

    for i in range(maxiter):
	# grad of beta
	grad = X.T.dot(X).dot(beta) - X.T.dot(y).reshape([p,1])
	# new estimate of beta
	beta_new = soft_thresh(beta-t*grad, t*lbd)
	obj_new = np.sum((y.reshape(beta.shape)-X.dot(beta_new))**2)/2 + lbd*np.sum(np.abs(beta_new))
	# check convergence
	if i >= miniter and np.abs(obj_new-obj) < tol :
	    break
	# store new variable
	beta = beta_new
	obj = obj_new
    return beta

def soft_thresh(beta, eta):
    return np.sign(beta) * np.max(np.abs(beta) - eta, 1).reshape(beta.shape)

def merge_two_mat(totalDelta,deltaMat):
    n = totalDelta.shape[0]
    m = totalDelta.shape[1]
    resultMat = np.zeros(totalDelta.shape)

    for nn in range(0,n):
	for mm in range(0,m):
    	    elementTotal = totalDelta[nn,mm]
            elementUpdate = deltaMat[nn,mm]
            if (elementTotal==0 or elementUpdate==0):
		resultMat[nn,mm]=elementTotal+elementUpdate
	    else:
		resultMat[nn,mm]=(elementTotal+elementUpdate)/(2.0)
    return resultMat


def solveBlockGlasso(signal):
    start = int(signal[0]) # include
    S_Matrix  = S_Matrix_bc.value
    W_matrix = W_Matrix_bc.value
    old_W = np.copy(W_matrix)
    end   = min(int(signal[1]),S_Matrix.shape[0]) # non-inclusive
    deltamatrix = np.zeros(S_Matrix.shape)
    NN = S_Matrix.shape[0]
    for n in range(start,end):
        W11 = np.delete(W_matrix,n,0)
        W11 = np.delete(W11,n,1)
        Z   = linalg.sqrtm(W11)

        s11 = S_Matrix[:,n]
        s11 = np.delete(s11,n)
        Y   = np.dot(nplinalg.inv(linalg.sqrtm(W11)),s11)
	Y = np.real(Y)
	Z = np.real(Z)
	B = lasso(Z,Y,beta_value)

    updated_column = np.dot(W11,B)

    matrix_ind = np.array(range(0,NN))
    matrix_ind = np.delete(matrix_ind,n)
    column_ind = 0
    for k in matrix_ind:
        deltamatrix[k,n]=updated_column[column_ind] - W_matrix[k,n]
        deltamatrix[n,k]=updated_column[column_ind] - W_matrix[k,n]
	    W_matrix[k,n] = updated_column[column_ind]
	    W_matrix[n,k] = updated_column[column_ind]
        column_ind = column_ind+1

    return W_matrix-old_W

if __name__ == '__main__':
    # read command line arguments
    num_workers = int(sys.argv[1])
    num_iterations = int(sys.argv[2])
    beta_value = float(sys.argv[3])
    input_filepath, output_filepath = sys.argv[4:]

    # create spark context
    conf = pyspark.SparkConf().setAppName("PGLasso").setMaster("local[{0}]".format(num_workers))
    sc = pyspark.SparkContext(conf=conf)

    # DONE: measure time starting here
    start_time = time.time() # time in seconds since the epoch as a floating point number

    # get dense covariance matrix from data file
    map_line = lambda s: np.fromstring(s, dtype = np.float64, sep=",")
    if os.path.isfile(input_filepath):
        # local file
        covariance_matrix = sc.textFile(input_filepath).map(map_line)
    else:
        # directory, or on HDFS
        rating_files = sc.wholeTextFiles(input_filepath)
        covariance_matrix = rating_files.flatMap(
            lambda pair: map_line(pair[1]))


    # get the dimension of the matrix
    dim = covariance_matrix.count()

    # convert covariance matrix in rdd format into numpy array, add penalty and stored in S_Matrix
    S_Matrix = np.array(covariance_matrix.collect())
    W_Matrix = S_Matrix + beta_value * np.eye(dim)
    old_W_Matrix = np.copy(W_Matrix)
    # remain covariance matrix fixed and never change it, broadcast to all workers
    S_Matrix_bc = sc.broadcast(S_Matrix)

    # determine strata block size.
    strata_size = np.ceil( float(dim) / float(num_workers))
    strata_size_bc = sc.broadcast(strata_size)

    # Here we are assigning each cell of the matrix to a worker
    covariance_matrix = covariance_matrix.zipWithIndex().map(lambda (row,idx):
        get_worker_id_for_position(idx)
        # partitionBy num_workers, by doing this we are distributing the
        # partitions of the RDD to all of the workers. Each worker gets one partition.
        # Lastly, we do a mapPartitionsWithIndex so each worker can group together
        # all cells that belong to the same block.
        # Make sure we preserve partitioning for correctness and parallelism efficiency
    ).partitionBy(num_workers)
    covariance_matrix = covariance_matrix \
      .mapPartitionsWithIndex(blockify_matrix, preservesPartitioning=True) \
      .cache()

    # finally, run PGlasso. Each iteration should update one strata.
    num_old_updates = 0
    totalDelta = np.zeros(S_Matrix.shape)
    for current_iteration in range(num_iterations):
        # perform updates for one strata in parallel
        # broadcast factor matrices to workers
        W_Matrix_bc = sc.broadcast(W_Matrix)

        updated = covariance_matrix.map(solveBlockGlasso, preservesPartitioning=True) \
            .collect()

        # unpersist outdated old factor matrices
        W_Matrix_bc.unpersist()
        for delta_matrix in updated:
            totalDelta = merge_two_mat(totalDelta,delta_matrix)
	    W_Matrix = old_W_Matrix + totalDelta
	    totalDelta = np.zeros(delta_matrix.shape)
        rmse,percent = calculate_loss(W_Matrix, old_W_Matrix)
        old_W_Matrix = np.copy(W_Matrix)
        # check convergence condition
        if(percent < TINY_EPS):
            print("converged!")
            break

    # print running time
    end_time = time.time()
    print('Running time is %f s' % (end_time - start_time))

    # Stop spark
    sc.stop()

    # DONE: print updated covariance matrix W to output_filepath
    np.savetxt(output_filepath,W_Matrix, fmt='%.5f', delimiter=",")
