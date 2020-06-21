package org.apache.spark.ml.sampling

import org.apache.spark.sql.{DataFrame, SparkSession}


/*

        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_grid (int): size of grid
            sigma (float): sigma of SOM
            learning_rate (float) learning rate of SOM
            n_iter (int): number of iterations
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

        N_inter= num_to_sample/2
        N_intra= num_to_sample/2

        # training SOM
        som= minisom.MiniSom(self.n_grid, self.n_grid, len(X[0]), sigma= self.sigma, learning_rate= self.learning_rate, random_seed= 3)
        som.train_random(X, self.n_iter)

        # constructing the grid
        grid_min= {}
        grid_maj= {}
        for i in range(len(y)):
            tmp= som.winner(X[i])
            idx= (tmp[0], tmp[1])
            if not idx in grid_min:
                grid_min[idx]= []
            if not idx in grid_maj:
                grid_maj[idx]= []
            if y[i] == self.minority_label:
                grid_min[idx].append(i)
            else:
                grid_maj[idx].append(i)

        # converting the grid to arrays
        for i in grid_min:
            grid_min[i]= np.array(grid_min[i])
        for i in grid_maj:
            grid_maj[i]= np.array(grid_maj[i])

        # filtering
        filtered= {}
        for i in grid_min:
            if not i in grid_maj:
                filtered[i]= True
            else:
                filtered[i]= (len(grid_maj[i]) + 1)/(len(grid_min[i])+1) < 1.0

        # computing densities
        densities= {}
        for i in filtered:
            if filtered[i]:
                if len(grid_min[i]) > 1:
                    densities[i]= len(grid_min[i])/np.mean(pairwise_distances(X[grid_min[i]]))**2
                else:
                    densities[i]= 10

        # all clusters can be filtered
        if len(densities) == 0:
            _logger.warning(self.__class__.__name__ + ": " +"all clusters filtered")
            return X.copy(), y.copy()

        # computing neighbour densities, using 4 neighborhood
        neighbors= [[0, 1], [0, -1], [1, 0], [-1, 0]]
        pair_densities= {}
        for i in densities:
            for n in neighbors:
                j= (i[0] + n[0], i[1] + n[1]),
                if j in densities:
                    pair_densities[(i,j)]= densities[i] + densities[j]

        # computing weights
        density_keys= list(densities.keys())
        density_vals= np.array(list(densities.values()))

        # determining pair keys and density values
        pair_keys= list(pair_densities.keys())
        pair_vals= np.array(list(pair_densities.values()))

        # determining densities
        density_vals= (1.0/density_vals)/np.sum(1.0/density_vals)
        pair_dens_vals= (1.0/pair_vals)/np.sum(1.0/pair_vals)

        # computing num of samples to generate
        if len(pair_vals) > 0:
            dens_num= N_intra
            pair_num= N_inter
        else:
            dens_num= N_inter + N_intra
            pair_num= 0

        # generating the samples according to the extracted distributions
        samples= []
        while len(samples) < dens_num:
            cluster_idx= density_keys[self.random_state.choice(np.arange(len(density_keys)), p= density_vals)]
            cluster= grid_min[cluster_idx]
            sample_a, sample_b= self.random_state.choice(cluster, 2)
            samples.append(self.sample_between_points(X[sample_a], X[sample_b]))

        while len(samples) < pair_num:
            idx= pair_keys[self.random_state.choice(np.arange(len(pair_keys)), p= pair_dens_vals)]
            cluster_a= grid_min[idx[0]]
            cluster_b= grid_min[idx[1]]
            samples.append(self.sample_between_points(X[self.random_state.choice(cluster_a)], X[self.random_state.choice(cluster_b)]))


 */

class SOMO {
  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    dfIn
  }

}
