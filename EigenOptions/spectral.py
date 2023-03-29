import numpy as np


def normalised_laplacian(adjacency_matrix, valency_matrix):
    laplacian_matrix = valency_matrix - adjacency_matrix
    raise_valency_power = np.isin(valency_matrix, 0)*1.5 - .5  # 1 if valency matrix was 0, -.5 otherwise
    sqrt_valency = np.power(valency_matrix, raise_valency_power)
    normalised_laplacian = np.dot(np.dot(sqrt_valency, laplacian_matrix), sqrt_valency)
    return normalised_laplacian


def eigen_decomposition(xx, num_eigenvalues):
    """
    calculate the eigenvalues and vectors of the transition matrix. Require policy to condense to just state transitions
    :param transition_matrix: state action transition matrix
    :return: eigenvalues, eigenvectors, unordered
    """
    eigenvalues, eigenvectors = np.linalg.eig(xx)
    eigenvectors = np.transpose(eigenvectors)
    top_eigenvalues, top_eigenvectors = top_n_eigen(num_eigenvalues, eigenvalues, eigenvectors)
    return top_eigenvalues, top_eigenvectors


def top_n_eigen(n, eigenvalues, eigenvectors):
    """
    return the top n eigenvalues and eigenvectors, ordered by eigenvalues
    """
    ordering = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[ordering][:n]
    sorted_eigenvectors = eigenvectors[ordering][:n]
    return sorted_eigenvalues, sorted_eigenvectors
