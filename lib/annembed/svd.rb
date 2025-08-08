# frozen_string_literal: true

require "numo/narray"

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
        matrix_array = prepare_matrix(matrix)
        
        u, s, v = randomized_svd_rust(matrix_array, k, n_iter)
        
        [convert_result(u), convert_result(s), convert_result(v)]
      end

      # Perform truncated SVD
      # @param matrix [Array, Numo::NArray] Input matrix
      # @param k [Integer] Number of components
      # @return [Array<Numo::NArray>] U, S, V matrices
      def truncated_svd(matrix, k)
        randomized_svd(matrix, k, n_iter: 2)
      end

      private

      def prepare_matrix(matrix)
        case matrix
        when Numo::NArray
          matrix
        when Array
          Numo::DFloat.cast(matrix)
        else
          raise ArgumentError, "Unsupported matrix type: #{matrix.class}"
        end
      end

      def convert_result(result)
        case result
        when Numo::NArray
          result
        when Array
          Numo::DFloat.cast(result)
        else
          result
        end
      end
    end
  end
end