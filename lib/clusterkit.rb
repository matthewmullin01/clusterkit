# frozen_string_literal: true

require_relative "clusterkit/version"
require_relative "clusterkit/clusterkit"
require_relative "clusterkit/configuration"

# Main module for ClusterKit gem
# Provides high-performance dimensionality reduction algorithms
module ClusterKit
  class Error < StandardError; end


  # Core error classes
  class DimensionError < Error; end
  class ConvergenceError < Error; end
  class InvalidParameterError < Error; end

  # Autoload classes for better performance
  autoload :UMAP, "clusterkit/umap"
  autoload :Utils, "clusterkit/utils"
  autoload :Preprocessing, "clusterkit/preprocessing"
  autoload :Silence, "clusterkit/silence"
  
  # SVD, PCA and Clustering need special handling - require them after the extension is loaded
  require_relative "clusterkit/svd"
  require_relative "clusterkit/pca"
  require_relative "clusterkit/clustering"

  class << self
    # Quick UMAP embedding
    # @param data [Array] Input data
    # @param n_components [Integer] Number of dimensions in output
    # @return [Array] Embedded data
    def umap(data, n_components: 2, **options)
      umap = UMAP.new(n_components: n_components, **options)
      umap.fit_transform(data)
    end

    # t-SNE is not yet implemented
    # @deprecated Not implemented - use UMAP instead
    def tsne(data, n_components: 2, **options)
      raise NotImplementedError, "t-SNE is not yet implemented. Please use UMAP instead, which provides similar dimensionality reduction capabilities."
    end

    # Estimate intrinsic dimension of data
    # @param data [Array, Numo::NArray] Input data
    # @param k [Integer] Number of neighbors to consider
    # @return [Float] Estimated intrinsic dimension
    def estimate_dimension(data, k: 10)
      Utils.estimate_intrinsic_dimension(data, k_neighbors: k)
    end

    # Perform randomized SVD
    # @param matrix [Array, Numo::NArray] Input matrix
    # @param k [Integer] Number of components
    # @return [Array<Numo::NArray>] U, S, V matrices
    def svd(matrix, k, n_iter: 2)
      SVD.randomized_svd(matrix, k, n_iter: n_iter)
    end
  end
end