# frozen_string_literal: true

module ClusterKit
  # Module for dimensionality reduction algorithms
  module Dimensionality
    # Load classes - can't use autoload with require issues
    require_relative "dimensionality/umap"
    require_relative "dimensionality/pca"
    require_relative "dimensionality/svd"
    
    # Module-level evaluation methods
    
    # Calculate reconstruction error for a dimensionality reduction
    # @param original_data [Array<Array<Numeric>>] Original high-dimensional data
    # @param reconstructed_data [Array<Array<Numeric>>] Reconstructed data
    # @return [Float] Mean squared reconstruction error
    def self.reconstruction_error(original_data, reconstructed_data)
      raise ArgumentError, "Data sizes don't match" if original_data.size != reconstructed_data.size
      
      total_error = 0.0
      original_data.zip(reconstructed_data).each do |orig, recon|
        error = orig.zip(recon).map { |o, r| (o - r) ** 2 }.sum
        total_error += error
      end
      
      total_error / original_data.size
    end
  end
end