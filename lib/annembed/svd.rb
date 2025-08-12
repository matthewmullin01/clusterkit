# frozen_string_literal: true

require_relative 'annembed_ruby'

module AnnEmbed
  # Module for SVD operations - methods are added by the Rust extension
  module SVD
    # Extend the module with Ruby convenience methods
    # The Rust extension already defined randomized_svd_rust on this module
    
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
    end
  end
end