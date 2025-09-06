# frozen_string_literal: true

require_relative '../clusterkit'
require_relative '../data_validator'

module ClusterKit
  module Dimensionality
    # Singular Value Decomposition
    # Decomposes a matrix into U, S, V^T components
    class SVD
      attr_reader :n_components, :n_iter, :random_seed
      attr_reader :u, :s, :vt, :n_features
      
      # Initialize a new SVD instance
      # @param n_components [Integer] Number of components to compute
      # @param n_iter [Integer] Number of iterations for randomized algorithm (default: 2)
      # @param random_seed [Integer, nil] Random seed for reproducibility
      def initialize(n_components: nil, n_iter: 2, random_seed: nil)
        @n_components = n_components
        @n_iter = n_iter
        @random_seed = random_seed
        @fitted = false
      end
      
      # Fit the model and transform data in one step
      # @param data [Array<Array<Numeric>>] Input data
      # @return [Array] Returns [U, S, Vt] matrices
      def fit_transform(data)
        validate_input(data)
        
        # Store data characteristics for later transform operations
        @n_features = data.first.size
        @original_data_id = data.object_id
        
        # Determine n_components if not set
        n_comp = @n_components || [data.size, data.first.size].min
        
        # Call the Rust implementation
        @u, @s, @vt = self.class.randomized_svd(data, n_comp, n_iter: @n_iter)
        @fitted = true
        
        [@u, @s, @vt]
      end
      
      # Fit the model to data
      # @param data [Array<Array<Numeric>>] Input data
      # @return [self]
      def fit(data)
        fit_transform(data)
        self
      end
      
      # Get the U matrix (left singular vectors)
      # @return [Array<Array<Float>>] U matrix
      def components_u
        raise RuntimeError, "Model must be fitted first" unless fitted?
        @u
      end
      
      # Get the singular values
      # @return [Array<Float>] Singular values
      def singular_values
        raise RuntimeError, "Model must be fitted first" unless fitted?
        @s
      end
      
      # Get the V^T matrix (right singular vectors, transposed)
      # @return [Array<Array<Float>>] V^T matrix
      def components_vt
        raise RuntimeError, "Model must be fitted first" unless fitted?
        @vt
      end
      
      # Check if the model has been fitted
      # @return [Boolean]
      def fitted?
        @fitted
      end
      
      # Transform data using fitted SVD (project onto components)
      # @param data [Array<Array<Numeric>>] Data to transform
      # @return [Array<Array<Float>>] Transformed data projected onto SVD components
      def transform(data)
        raise RuntimeError, "Model must be fitted first" unless fitted?
        validate_transform_input(data)
        
        if data.object_id == @original_data_id
          # Same data that was fitted - return U * S
          @u.map.with_index do |row, i|
            row.map.with_index { |val, j| val * @s[j] }
          end
        else
          # New data - project onto V components: data × V
          # Since we have V^T, we need to transpose it back to V
          # V = V^T^T, so we project: data × V^T^T
          transform_new_data(data)
        end
      end
      
      # Inverse transform (reconstruct from components)
      # @param transformed_data [Array<Array<Float>>] Transformed data
      # @return [Array<Array<Float>>] Reconstructed data
      def inverse_transform(transformed_data)
        raise RuntimeError, "Model must be fitted first" unless fitted?
        
        # Reconstruction: (U * S) * V^T
        # transformed_data should be U * S
        # We multiply by V^T to reconstruct
        
        result = []
        transformed_data.each do |row|
          reconstructed = Array.new(@vt.first.size, 0.0)
          row.each_with_index do |val, i|
            @vt[i].each_with_index do |v, j|
              reconstructed[j] += val * v
            end
          end
          result << reconstructed
        end
        result
      end
      
      # Class method for randomized SVD (kept for compatibility)
      # @param matrix [Array<Array<Numeric>>] Input matrix
      # @param k [Integer] Number of components
      # @param n_iter [Integer] Number of iterations
      # @return [Array] Returns [U, S, Vt]
      def self.randomized_svd(matrix, k, n_iter: 2)
        ::ClusterKit::SVD.randomized_svd_rust(matrix, k, n_iter)
      end
      
      private
      
      def validate_input(data)
        DataValidator.validate_standard(data, check_finite: false)
      end
      
      def validate_transform_input(data)
        DataValidator.validate_standard(data, check_finite: false)
        
        # Check feature count matches training data
        if data.first.size != @n_features
          raise ArgumentError, "New data has #{data.first.size} features, but model was fitted with #{@n_features} features"
        end
      end
      
      # Transform new data by projecting onto V components
      # Mathematical operation: new_data × V, where V = V^T^T
      def transform_new_data(data)
        # V^T is stored as @vt (shape: n_components × n_features)
        # We need V (shape: n_features × n_components)
        # V = V^T^T, so we transpose @vt
        
        result = []
        data.each do |sample|
          # Project sample onto each component (column of V = row of V^T)
          projected = Array.new(@vt.size, 0.0)
          
          @vt.each_with_index do |vt_row, comp_idx|
            # Dot product: sample · vt_row (this is sample · V[:, comp_idx])
            dot_product = 0.0
            sample.each_with_index do |val, feat_idx|
              dot_product += val * vt_row[feat_idx]
            end
            projected[comp_idx] = dot_product
          end
          
          result << projected
        end
        
        result
      end
    end
  end
end