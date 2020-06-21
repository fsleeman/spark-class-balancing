package org.apache.spark.ml.sampling

import org.apache.spark.sql.{DataFrame, SparkSession}

/*
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

        def taxicab_sample(n, r):
            sample = []
            random_numbers= self.random_state.rand(n)

            for i in range(n):
                #spread = r - np.sum(np.abs(sample))
                spread= r
                if len(sample) > 0:
                    spread-= abs(sample[-1])
                sample.append(spread * (2 * random_numbers[i] - 1))

            return self.random_state.permutation(sample)

        minority= X[y == self.minority_label]
        majority= X[y == self.majority_label]

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances= pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point= minority[i]
            remaining_energy= energy
            r= 0.0
            sorted_distances= np.argsort(distances[i])
            current_majority= 0

            while True:
                if current_majority > len(majority):
                    break

                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change= remaining_energy / (current_majority + 1.0)
                    else:
                        radius_change= remaining_energy / current_majority

                    r+= radius_change
                    break

                radius_change= remaining_energy / (current_majority + 1.0)

                if distances[i, sorted_distances[current_majority]] >= r + radius_change:
                    r+= radius_change
                    break
                else:
                    if current_majority == 0:
                        last_distance= 0.0
                    else:
                        last_distance= distances[i, sorted_distances[current_majority - 1]]

                    radius_change= distances[i, sorted_distances[current_majority]] - last_distance
                    r+= radius_change
                    remaining_energy-= radius_change * (current_majority + 1.0)
                    current_majority+= 1

            radii[i] = r

            for j in range(current_majority):
                majority_point= majority[sorted_distances[j]].astype(float)
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    majority_point+= (1e-6 * self.random_state.rand(len(majority_point)) + 1e-6) * self.random_state.choice([-1.0, 1.0], len(majority_point))
                    d = np.sum(np.abs(minority_point - majority_point))

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority= majority.astype(float)
        majority += translations

        appended= []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * num_to_sample))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point + taxicab_sample(len(minority_point), r))

        if len(appended) == 0:
            _logger.info("No samples were added")
            return X.copy(), y.copy()




 */



class CCR {

  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    dfIn
  }

}
