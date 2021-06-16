# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This is a kmeans reimplementation that follows the torch unsup package
# See https://github.com/koraykv/unsup
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def kmeans(x, k, niter=1, batchsize=1000):
    batchsize = min(batchsize, x.shape[0])

    nsamples = x.shape[0]
    ndims = x.shape[1]

    x2 = np.sum(x ** 2, axis=1)
    centroids = np.random.randn(k, ndims)
    centroidnorm = np.sqrt(np.sum(centroids ** 2, axis=1, keepdims=True))
    centroids = centroids / centroidnorm
    totalcounts = np.zeros(k)

    for i in range(niter):
        c2 = np.sum(centroids ** 2, axis=1, keepdims=True) * 0.5
        summation = np.zeros((k, ndims))
        counts = np.zeros(k)
        loss = 0

        for j in range(0, nsamples, batchsize):
            lastj = min(j + batchsize, nsamples)
            batch = x[j:lastj]
            m = batch.shape[0]

            tmp = np.dot(centroids, batch.T)
            tmp = tmp - c2
            val = np.max(tmp, 0)
            labels = np.argmax(tmp, 0)
            loss = loss + np.sum(np.sum(x2[j:lastj]) * 0.5 - val)

            S = np.zeros((k, m))
            S[labels, np.arange(m)] = 1
            summation = summation + np.dot(S, batch)
            counts = counts + np.sum(S, axis=1)

        for j in range(k):
            if counts[j] > 0:
                centroids[j] = summation[j] / counts[j]

        totalcounts = totalcounts + counts
        for j in range(k):
            if totalcounts[j] == 0:
                idx = np.random.choice(nsamples)
                centroids[j] = x[idx]

    return centroids

def cluster_feats(filehandle, base_classes, cachefile, n_clusters=100):
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as f:
            centroids = pkl.load(f)
    else:
        centroids = []
        all_labels = filehandle['all_labels'][...]
        all_feats = filehandle['all_feats']

        count = filehandle['count'][0]
        for j, i in enumerate(base_classes):
            print('Clustering class {:d}:{:d}'.format(j, i))
            idx = np.where(all_labels == i)[0]
            idx = idx[idx < count]
            X = all_feats[idx, :]

            centroids_this = kmeans(X, n_clusters, 20)
            centroids.append(centroids_this)
        with open(cachefile, 'wb') as f:
            pkl.dump(centroids, f)
    return centroids

def get_difference_vectors(c_i):
    diff_i = c_i[:, np.newaxis, :] - c_i[np.newaxis, :, :]
    diff_i = diff_i.reshape((-1, diff_i.shape[2]))
    diff_i_norm = np.sqrt(np.sum(diff_i ** 2, axis=1, keepdims=True))
    diff_i = diff_i / (diff_i_norm + 0.00001)
    return diff_i

def mine_analogies(centroids):
    n_clusters = centroids[0].shape[0]

    analogies = np.zeros((n_clusters * n_clusters * len(centroids), 4), dtype=int)
    analogy_scores = np.zeros(analogies.shape[0])
    start = 0

    I, J = np.unravel_index(np.arange(n_clusters ** 2), (n_clusters, n_clusters))
    # for every class
    for i, c_i in enumerate(centroids):

        # get normalized difference vectors between cluster centers
        diff_i = get_difference_vectors(c_i)
        diff_i_t = torch.tensor(diff_i, device=device)


        bestdots = np.zeros(diff_i.shape[0])
        bestdots_idx = np.zeros((diff_i.shape[0], 2), dtype=int)

        # for every other class
        for j, c_j in enumerate(centroids):
            if i == j:
                continue
            print(i, j)

            # get normalized difference vectors
            diff_j = get_difference_vectors(c_j)
            diff_j = torch.tensor(diff_j, device=device)

            # compute cosine distance and take the maximum
            dots = diff_i_t.mm(diff_j.transpose(0, 1))
            maxdots, argmaxdots = dots.max(1)
            maxdots = maxdots.cpu().numpy().reshape(-1)
            argmaxdots = argmaxdots.cpu().numpy().reshape(-1)

            # if maximum is better than best seen so far, update
            better_idx = maxdots > bestdots
            bestdots[better_idx] = maxdots[better_idx]
            bestdots_idx[better_idx, 0] = j * n_clusters + I[argmaxdots[better_idx]]
            bestdots_idx[better_idx, 1] = j * n_clusters + J[argmaxdots[better_idx]]


        # store discovered analogies
        stop = start + diff_i.shape[0]
        analogies[start:stop, 0] = i * n_clusters + I
        analogies[start:stop, 1] = i * n_clusters + J
        analogies[start:stop, 2:] = bestdots_idx
        analogy_scores[start:stop] = bestdots
        start = stop

    # prune away trivial analogies
    good_analogies = (analogy_scores > 0) & (analogies[:, 0] != analogies[:, 1]) & (analogies[:, 2] != analogies[:, 3])
    return analogies[good_analogies, :], analogy_scores[good_analogies]