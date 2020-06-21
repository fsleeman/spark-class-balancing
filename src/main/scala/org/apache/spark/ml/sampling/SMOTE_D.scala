package org.apache.spark.ml.sampling

import org.apache.spark.sql.{DataFrame, SparkSession}

/*

    """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            k (int): number of neighbors in nearest neighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """


    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])

        if num_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min= X[y == self.minority_label]

        # fitting nearest neighbors model
        nn= NearestNeighbors(n_neighbors= min([len(X_min), self.k+1]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)

        # extracting standard deviations of distances
        stds= np.std(dist[:,1:], axis= 1)
        # estimating sampling density
        if np.sum(stds) > 0:
            p_i= stds/np.sum(stds)
        else:
            _logger.warning(self.__class__.__name__ + ": " + "zero distribution")
            return X.copy(), y.copy()

        # the other component of sampling density
        p_ij= dist[:,1:]/np.sum(dist[:,1:], axis= 1)[:,None]

        # number of samples to generate between minority points
        counts_ij= num_to_sample*p_i[:,None]*p_ij

        # do the sampling
        samples= []
        for i in range(len(p_i)):
            for j in range(min([len(X_min)-1, self.k])):
                while counts_ij[i][j] > 0:
                    if self.random_state.random_sample() < counts_ij[i][j]:
                        samples.append(X_min[i] + (X_min[ind[i][j+1]] - X_min[i])/(counts_ij[i][j]+1))
                    counts_ij[i][j]= counts_ij[i][j] - 1

        if len(samples) > 0:
            return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        else:
            return X.copy(), y.copy()

 */


class SMOTE_D {
  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    dfIn
  }

}
