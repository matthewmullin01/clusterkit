# frozen_string_literal: true

require "numo/narray"

module AnnEmbed
  # Data preprocessing utilities
  module Preprocessing
    class << self
      # Normalize data using specified method
      # @param data [Array, Numo::NArray] Input data
      # @param method [Symbol] Normalization method (:standard, :minmax, :l2)
      # @return [Numo::NArray] Normalized data
      def normalize(data, method: :standard)
        data_array = prepare_data(data)
        
        case method
        when :standard
          standard_normalize(data_array)
        when :minmax
          minmax_normalize(data_array)
        when :l2
          l2_normalize(data_array)
        else
          raise ArgumentError, "Unknown normalization method: #{method}"
        end
      end

      # Reduce dimensionality using PCA before embedding
      # @param data [Array, Numo::NArray] Input data
      # @param n_components [Integer] Number of PCA components
      # @return [Numo::NArray] Reduced data
      def pca_reduce(data, n_components)
        data_array = prepare_data(data)
        
        # Use SVD for PCA
        mean = data_array.mean(axis: 0)
        centered = data_array - mean
        
        u, s, vt = SVD.randomized_svd(centered, n_components)
        u * s
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

      def standard_normalize(data)
        mean = data.mean(axis: 0)
        std = data.stddev(axis: 0)
        std[std.eq(0)] = 1.0 # Avoid division by zero
        
        (data - mean) / std
      end

      def minmax_normalize(data)
        min = data.min(axis: 0)
        max = data.max(axis: 0)
        range = max - min
        range[range.eq(0)] = 1.0 # Avoid division by zero
        
        (data - min) / range
      end

      def l2_normalize(data)
        norms = Numo::NMath.sqrt((data**2).sum(axis: 1))
        norms[norms.eq(0)] = 1.0 # Avoid division by zero
        
        data / norms.expand_dims(1)
      end
    end
  end
end