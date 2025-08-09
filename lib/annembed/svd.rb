# frozen_string_literal: true

# Pure Ruby wrapper for SVD operations

module AnnEmbed
  # Module for SVD operations
  module SVD
    class << self
      # Perform randomized SVD
      # @param matrix [Array, Numo::NArray] Input matrix
      # @param k [Integer] Number of components
      # @param n_iter [Integer] Number of iterations
      # @return [Array<Numo::NArray>] U, S, V matrices
      def randomized_svd(matrix, k, n_iter: 2)
        raise ArgumentError, "Unsupported matrix type: #{matrix.class}" unless matrix.is_a?(Array)
        
        randomized_svd_rust(matrix, k, n_iter)
      end

      # Perform truncated SVD
      # @param matrix [Array, Numo::NArray] Input matrix
      # @param k [Integer] Number of components
      # @return [Array<Numo::NArray>] U, S, V matrices
      def truncated_svd(matrix, k)
        randomized_svd(matrix, k, n_iter: 2)
      end

      private
    end
  end
end