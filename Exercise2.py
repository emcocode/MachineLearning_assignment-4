import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

div_zero = 1e-6 # We use this number to replace denominators that are too small

# Sammon stress function
def sammonStress(dist_in, dist_out):
    numerator = (dist_out - dist_in) ** 2
    denominator = dist_in.copy()
    denominator[denominator < div_zero] = div_zero

    cost = np.sum(numerator / denominator) / np.sum(dist_in)
    return cost

# Sammon mapping
def sammon(X, iterations, threshold, learning_rate):
    # Normalize and create randomized output space
    X = MinMaxScaler().fit_transform(X)
    n = X.shape[0]
    Y = np.random.rand(n, 2)

    # Calculate distances, stress and sum
    dist_in, dist_out = euclidean_distances(X), euclidean_distances(Y)
    iter = 0 
    stress = sammonStress(dist_in, dist_out)
    c = np.sum(dist_in)

    # Iterate only during following conditions
    while ((iter < iterations) & (threshold <= stress)):
        # Calculate distances and stress
        iter += 1
        dist_out = euclidean_distances(Y)
        stress = sammonStress(dist_in, dist_out)
        print(f"iteration {iter} and stress: {stress}") # Makes the progress easier to track, and also to find the right thresholds/learning rates
        
        for i in range(X.shape[0]):
            # Reshape rows
            yi = Y[i].reshape((1, 2))
            xi = X[i].reshape((1, X.shape[1]))

            # Create a copy of X and Y, and remove row i from copied array
            yj = np.delete(Y.copy(), i, axis = 0)
            xj = np.delete(X.copy(), i, axis = 0)
            
            # Calculate and reshape distances
            space_out = euclidean_distances(yi, yj).reshape((yj.shape[0], 1))
            space_in = euclidean_distances(xi, xj).reshape((xj.shape[0], 1))
            
            # Calculate numerator and denominator (making sure den. is not too small)
            numerator = space_in - space_out
            denominator = space_out * space_in
            denominator[denominator < div_zero] = div_zero
            
            # Calculate partial derivatives
            first = (-2 * np.sum((numerator * (yi - yj) / denominator), axis = 0) / c)
            squared = ((yi - yj) ** 2) / space_out
            last_part = (1 + ((yi - yj) / space_out))
            second = (-2 * np.sum(((yi - yj) - squared * last_part) / denominator, axis = 0) / c)
            
            # Update Y[i]
            Y[i] = Y[i] - learning_rate * (first / abs(second))
    return Y
