# frozen_string_literal: true

require 'fileutils'
require 'json'
require_relative '../configuration'
require_relative '../silence'

module ClusterKit
  module Dimensionality
    class UMAP
    attr_reader :n_components, :n_neighbors, :random_seed, :nb_grad_batch, :nb_sampling_by_edge
    
    # Initialize a new UMAP instance
    # @param n_components [Integer] Target number of dimensions (default: 2)
    # @param n_neighbors [Integer] Number of neighbors for manifold approximation (default: 15)
    # @param random_seed [Integer, nil] Random seed for reproducibility (default: nil)
    # @param nb_grad_batch [Integer] Number of gradient descent batches (default: 10)
    #                                Controls training iterations - lower = faster but less accurate
    # @param nb_sampling_by_edge [Integer] Number of negative samples per edge (default: 8)
    #                                      Controls sampling quality - lower = faster but less accurate
    def initialize(n_components: 2, n_neighbors: 15, random_seed: nil, 
                   nb_grad_batch: 10, nb_sampling_by_edge: 8)
      @n_components = n_components
      @n_neighbors = n_neighbors
      @random_seed = random_seed
      @nb_grad_batch = nb_grad_batch
      @nb_sampling_by_edge = nb_sampling_by_edge
      @fitted = false
      # Don't create RustUMAP yet - will be created in fit/fit_transform with adjusted parameters
      @rust_umap = nil
    end
    
    # Fit the model to the data (training)
    # @param data [Array<Array<Numeric>>] Training data as 2D array
    # @return [self] Returns self for method chaining
    # @note UMAP's training process inherently produces embeddings. Since the
    #       underlying Rust implementation doesn't separate training from 
    #       transformation, we call fit_transform but discard the embeddings.
    #       Use fit_transform if you need both training and the transformed data.
    def fit(data)
      validate_input(data)
      
      # Always recreate RustUMAP for fit to ensure fresh fit
      @rust_umap = nil
      create_rust_umap_with_adjusted_params(data)
      
      # UMAP doesn't separate training from transformation internally,
      # so we call fit_transform but discard the result
      begin
        Silence.maybe_silence do
          @rust_umap.fit_transform(data)
        end
        @fitted = true
        self
      rescue StandardError => e
        handle_umap_error(e, data)
      rescue => e
        # Handle fatal errors that aren't StandardError
        handle_umap_error(RuntimeError.new(e.message), data)
      end
    end
    
    # Transform data using the fitted model
    # @param data [Array<Array<Numeric>>] Data to transform
    # @return [Array<Array<Float>>] Transformed data in reduced dimensions
    # @raise [RuntimeError] If model hasn't been fitted yet
    def transform(data)
      raise RuntimeError, "Model must be fitted before transform. Call fit or fit_transform first." unless fitted?
      validate_input(data, check_min_samples: false)
      Silence.maybe_silence do
        @rust_umap.transform(data)
      end
    end
    
    # Fit the model and transform the data in one step
    # @param data [Array<Array<Numeric>>] Training data as 2D array
    # @return [Array<Array<Float>>] Transformed data in reduced dimensions
    def fit_transform(data)
      validate_input(data)
      
      # Always recreate RustUMAP for fit_transform to ensure fresh fit
      @rust_umap = nil
      create_rust_umap_with_adjusted_params(data)
      
      begin
        result = Silence.maybe_silence do
          @rust_umap.fit_transform(data)
        end
        @fitted = true
        result
      rescue StandardError => e
        handle_umap_error(e, data)
      rescue => e
        # Handle fatal errors that aren't StandardError
        handle_umap_error(RuntimeError.new(e.message), data)
      end
    end
    
    # Check if the model has been fitted
    # @return [Boolean] true if model is fitted, false otherwise
    def fitted?
      @fitted
    end
    
    # Save the fitted model to a file
    # @param path [String] Path where to save the model
    # @raise [RuntimeError] If model hasn't been fitted yet
    def save(path)
      raise RuntimeError, "No model to save. Call fit or fit_transform first." unless fitted?
      
      # Ensure directory exists
      dir = File.dirname(path)
      FileUtils.mkdir_p(dir) unless dir == '.' || dir == '/'
      
      @rust_umap.save_model(path)
    end
    
    # Load a fitted model from a file
    # @param path [String] Path to the saved model
    # @return [UMAP] A new UMAP instance with the loaded model
    # @raise [ArgumentError] If file doesn't exist
    def self.load(path)
      raise ArgumentError, "File not found: #{path}" unless File.exist?(path)
      
      # Load the Rust model
      rust_umap = ::ClusterKit::RustUMAP.load_model(path)
      
      # Create a new UMAP instance with the loaded model
      instance = allocate
      instance.instance_variable_set(:@rust_umap, rust_umap)
      instance.instance_variable_set(:@fitted, true)
      # The model file should contain these parameters, but for now we don't have access
      instance.instance_variable_set(:@n_components, nil)
      instance.instance_variable_set(:@n_neighbors, nil)
      instance.instance_variable_set(:@random_seed, nil)
      
      instance
    end
    
    # Export transformed data to JSON (utility method for caching)
    # @param data [Array<Array<Float>>] Transformed data to export
    # @param path [String] Path where to save the data
    def self.export_data(data, path)
      File.write(path, JSON.pretty_generate(data))
    end
    
    # Import transformed data from JSON (utility method for caching)
    # @param path [String] Path to the saved data
    # @return [Array<Array<Float>>] The loaded data
    def self.import_data(path)
      JSON.parse(File.read(path))
    end
    
    private
    
    def handle_umap_error(error, data)
      error_msg = error.message
      n_samples = data.size
      
      case error_msg
      when /isolated point/i, /graph will not be connected/i
        raise ::ClusterKit::IsolatedPointError, <<~MSG
          UMAP found isolated points in your data that are too far from other points.
          
          This typically happens when:
          • Your data contains outliers that are very different from other points
          • You're using random data without inherent structure
          • The n_neighbors parameter (#{@n_neighbors}) is too high for your data distribution
          
          Solutions:
          1. Reduce n_neighbors (try 5 or even 3): UMAP.new(n_neighbors: 5)
          2. Remove outliers from your data before applying UMAP
          3. Ensure your data has some structure (not purely random)
          4. For small datasets (< 50 points), consider using PCA instead
          
          Your data: #{n_samples} samples, #{data.first&.size || 0} dimensions
        MSG
        
      when /assertion failed.*box_size/i
        raise ::ClusterKit::ConvergenceError, <<~MSG
          UMAP failed to converge due to numerical instability in your data.
          
          This typically happens when:
          • Data points are too spread out or have extreme values
          • The scale of different features varies wildly
          • There are duplicate or nearly-duplicate points
          
          Solutions:
          1. Normalize your data first: ClusterKit::Preprocessing.normalize(data)
          2. Use a smaller n_neighbors value: UMAP.new(n_neighbors: 5)  
          3. Check for and remove duplicate points
          4. Scale your data to a reasonable range (e.g., 0-1 or -1 to 1)
          
          Your data: #{n_samples} samples, #{data.first&.size || 0} dimensions
        MSG
        
      when /n_neighbors.*larger than/i, /too many neighbors/i
        raise ::ClusterKit::InvalidParameterError, <<~MSG
          The n_neighbors parameter (#{@n_neighbors}) is too large for your dataset size (#{n_samples}).
          
          UMAP needs n_neighbors to be less than the number of samples.
          Suggested value: #{[5, (n_samples * 0.1).to_i].max}
          
          This should have been auto-adjusted. If you're seeing this error, please report it.
        MSG
        
      else
        # For unknown errors, still provide some guidance
        raise ::ClusterKit::Error, <<~MSG
          UMAP encountered an error: #{error_msg}
          
          Common solutions:
          1. Try reducing n_neighbors (current: #{@n_neighbors})
          2. Normalize your data first
          3. Check for NaN or infinite values in your data
          4. Ensure you have at least 10 data points
          
          If this persists, consider using PCA for dimensionality reduction instead.
        MSG
      end
    end
    
    def validate_input(data, check_min_samples: true)
      raise ArgumentError, "Input must be an array" unless data.is_a?(Array)
      raise ArgumentError, "Input cannot be empty" if data.empty?
      
      first_row = data.first
      raise ArgumentError, "Input must be a 2D array (array of arrays)" unless first_row.is_a?(Array)
      
      row_length = first_row.length
      min_val = Float::INFINITY
      max_val = -Float::INFINITY
      
      # First validate data structure and types
      data.each_with_index do |row, i|
        unless row.is_a?(Array)
          raise ArgumentError, "Row #{i} is not an array"
        end
        
        if row.length != row_length
          raise ArgumentError, "All rows must have the same length (row #{i} has #{row.length} elements, expected #{row_length})"
        end
        
        row.each_with_index do |val, j|
          unless val.is_a?(Numeric)
            raise ArgumentError, "Element at position [#{i}, #{j}] is not numeric"
          end
          
          # Only check for NaN/Infinite on floats
          if val.is_a?(Float) && (val.nan? || val.infinite?)
            raise ArgumentError, "Element at position [#{i}, #{j}] is NaN or Infinite"
          end
          
          # Track data range
          val_f = val.to_f
          min_val = val_f if val_f < min_val
          max_val = val_f if val_f > max_val
        end
      end
      
      # Check for sufficient data points after validating structure (only for fit operations)
      if check_min_samples && data.size < 10
        raise ::ClusterKit::InsufficientDataError, <<~MSG
          UMAP requires at least 10 data points, but only #{data.size} provided.
          
          For small datasets, consider:
          1. Using PCA instead: ClusterKit::Dimensionality::PCA.new(n_components: 2)
          2. Collecting more data points
          3. Using simpler visualization methods
        MSG
      end
      
      # Check for extreme data ranges that might cause numerical issues
      data_range = max_val - min_val
      if data_range > 1000
        warn "WARNING: Large data range detected (#{data_range.round(2)}). Consider normalizing your data to prevent numerical instability."
      end
    end
    
    def create_rust_umap_with_adjusted_params(data)
      # Only create if not already created
      return if @rust_umap
      
      n_samples = data.size
      
      # Automatically adjust n_neighbors if it's too high for the dataset
      # n_neighbors should be less than n_samples
      # Use a reasonable default: min(15, n_samples / 4) but at least 2
      max_neighbors = [n_samples - 1, 2].max  # At least 2, but less than n_samples
      suggested_neighbors = [[15, n_samples / 4].min.to_i, 2].max
      
      adjusted_n_neighbors = @n_neighbors
      if @n_neighbors > max_neighbors
        adjusted_n_neighbors = [suggested_neighbors, max_neighbors].min
        
        if ::ClusterKit.configuration.verbose
          warn "UMAP: Adjusted n_neighbors from #{@n_neighbors} to #{adjusted_n_neighbors} for dataset with #{n_samples} samples"
        end
      end
      
      @rust_umap = ::ClusterKit::RustUMAP.new({
        n_components: @n_components,
        n_neighbors: adjusted_n_neighbors,
        random_seed: @random_seed,
        nb_grad_batch: @nb_grad_batch,
        nb_sampling_by_edge: @nb_sampling_by_edge
      })
    end
    end
  end
end