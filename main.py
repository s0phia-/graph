from Env.gridWorld import GridWorld
from EigenOptions.spectral import eigen_decomposition, normalised_laplacian
from EigenOptions.clustering import eigen_clusters
from EigenOptions.visualisation import plot_clusters, plot_eigenvector
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create gridworld environment
    grid = GridWorld()

    # calculate the state action transition matrix for the gridworld
    adjacency_matrix = grid.adjacency_matrix
    valency_matrix = grid.valency_matrix
    normalised_laplacian = normalised_laplacian(adjacency_matrix, valency_matrix)

    # find the eigenvalues and vectors for the state transition matrix
    eigenvalues, eigenvectors = eigen_decomposition(normalised_laplacian, 10)

    # plot_eigenvector(eigenvectors[0], grid.grid)
    # plot_eigenvector(eigenvectors[1], grid.grid)
    # plot_eigenvector(eigenvectors[2], grid.grid)
    # plot_eigenvector(eigenvectors[3], grid.grid)
    # plot_eigenvector(eigenvectors[4], grid.grid)
    # plot_eigenvector(eigenvectors[5], grid.grid)

    # find inertia values for different cluster sizes. I used the top k eigenvectors for k clusters
    inertias = []
    K = []
    for k in range(1,10):
        cluster_labels, inertia = eigen_clusters(k, eigenvectors[:k])
        plot_clusters(cluster_labels, grid.grid)
        print(str(k) + str(inertia))
        inertias.append(inertia)
        K.append(k)
    print(K, inertias)
    plt.plot(K, inertias, 'bx-')
    plt.show()

    # go straight to x clusters
    # cluster_labels = eigen_clusters(6, eigenvectors)
