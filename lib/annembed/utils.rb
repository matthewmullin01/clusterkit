# frozen_string_literal: true

require "numo/narray"

module Annembed
  # Utility functions for data analysis
  module Utils
    class << self
      # Estimate the intrinsic dimension of data
      # @param data [Array, Numo::NArray] Input data
      # @param k_neighbors [Integer] Number of neighbors to consider
      # @return [Float] Estimated intrinsic dimension
      def estimate_intrinsic_dimension(data, k_neighbors: 10)
        data_array = prepare_data(data)
        
        estimate_intrinsic_dimension_rust(data_array, k_neighbors)
      end

      # Estimate hubness in the data
      # @param data [Array, Numo::NArray] Input data
      # @return [Hash] Hubness statistics
      def estimate_hubness(data)
        data_array = prepare_data(data)
        
        result = estimate_hubness_rust(data_array)
        symbolize_keys(result)
      end

      # Measure neighborhood stability through embedding
      # @param original_data [Array, Numo::NArray] Original high-dimensional data
      # @param embedded_data [Array, Numo::NArray] Embedded low-dimensional data
      # @param k [Integer] Number of neighbors to check
      # @return [Float] Stability score (0-1, higher is better)
      def neighborhood_stability(original_data, embedded_data, k: 15)
        orig_array = prepare_data(original_data)
        embed_array = prepare_data(embedded_data)
        
        # TODO: Implement neighborhood stability calculation
        raise NotImplementedError, "Neighborhood stability not implemented yet"
      end

      private

      def prepare_data(data)
        case data
        when Numo::NArray
          data
        when Array
          Numo::DFloat.cast(data)
        else
          raise ArgumentError, "Unsupported data type: #{data.class}"
        end
      end

      def symbolize_keys(hash)
        return hash unless hash.is_a?(Hash)
        
        hash.transform_keys(&:to_sym)
      end
    end
  end
end