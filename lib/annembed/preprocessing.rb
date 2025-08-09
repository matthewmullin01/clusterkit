# frozen_string_literal: true

# Pure Ruby implementation of preprocessing functions

module AnnEmbed
  # Data preprocessing utilities
  module Preprocessing
    class << self
      # Normalize data using specified method
      # @param data [Array] Input data (2D array)
      # @param method [Symbol] Normalization method (:standard, :minmax, :l2)
      # @return [Array] Normalized data
      def normalize(data, method: :standard)
        raise ArgumentError, "Unsupported data type: #{data.class}" unless data.is_a?(Array)
        
        case method
        when :standard
          standard_normalize(data)
        when :minmax
          minmax_normalize(data)
        when :l2
          l2_normalize(data)
        else
          raise ArgumentError, "Unknown normalization method: #{method}"
        end
      end

      # Reduce dimensionality using PCA before embedding
      # @param data [Array] Input data
      # @param n_components [Integer] Number of PCA components
      # @return [Array] Reduced data
      def pca_reduce(data, n_components)
        # Note: This would require SVD implementation in pure Ruby
        # For now, raise an error suggesting to use the Rust-based SVD module
        raise NotImplementedError, "PCA reduction requires the SVD module which needs to be called directly"
      end

      private

      def standard_normalize(data)
        # Pure Ruby implementation of standard normalization
        return data if data.empty?
        
        # Calculate mean and std for each column
        n_rows = data.size
        n_cols = data.first.size
        
        means = Array.new(n_cols, 0.0)
        stds = Array.new(n_cols, 0.0)
        
        # Calculate means
        data.each do |row|
          row.each_with_index { |val, j| means[j] += val }
        end
        means.map! { |m| m / n_rows }
        
        # Calculate standard deviations
        data.each do |row|
          row.each_with_index { |val, j| stds[j] += (val - means[j]) ** 2 }
        end
        stds.map! { |s| Math.sqrt(s / n_rows) }
        stds.map! { |s| s == 0 ? 1.0 : s } # Avoid division by zero
        
        # Normalize
        data.map do |row|
          row.map.with_index { |val, j| (val - means[j]) / stds[j] }
        end
      end

      def minmax_normalize(data)
        # Pure Ruby implementation of min-max normalization
        return data if data.empty?
        
        n_cols = data.first.size
        mins = Array.new(n_cols) { Float::INFINITY }
        maxs = Array.new(n_cols) { -Float::INFINITY }
        
        # Find min and max for each column
        data.each do |row|
          row.each_with_index do |val, j|
            mins[j] = val if val < mins[j]
            maxs[j] = val if val > maxs[j]
          end
        end
        
        # Calculate ranges
        ranges = mins.zip(maxs).map { |min, max| max - min }
        ranges.map! { |r| r == 0 ? 1.0 : r } # Avoid division by zero
        
        # Normalize
        data.map do |row|
          row.map.with_index { |val, j| (val - mins[j]) / ranges[j] }
        end
      end

      def l2_normalize(data)
        # Pure Ruby implementation of L2 normalization
        data.map do |row|
          norm = Math.sqrt(row.sum { |val| val ** 2 })
          norm = 1.0 if norm == 0 # Avoid division by zero
          row.map { |val| val / norm }
        end
      end
    end
  end
end