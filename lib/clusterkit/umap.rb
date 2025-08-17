# frozen_string_literal: true

require 'fileutils'
require 'json'
require_relative 'configuration'
require_relative 'silence'

module ClusterKit
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
      
      # Create RustUMAP with adjusted parameters if needed
      create_rust_umap_with_adjusted_params(data)
      
      # UMAP doesn't separate training from transformation internally,
      # so we call fit_transform but discard the result
      Silence.maybe_silence do
        @rust_umap.fit_transform(data)
      end
      @fitted = true
      self
    end
    
    # Transform data using the fitted model
    # @param data [Array<Array<Numeric>>] Data to transform
    # @return [Array<Array<Float>>] Transformed data in reduced dimensions
    # @raise [RuntimeError] If model hasn't been fitted yet
    def transform(data)
      raise RuntimeError, "Model must be fitted before transform. Call fit or fit_transform first." unless fitted?
      validate_input(data)
      Silence.maybe_silence do
        @rust_umap.transform(data)
      end
    end
    
    # Fit the model and transform the data in one step
    # @param data [Array<Array<Numeric>>] Training data as 2D array
    # @return [Array<Array<Float>>] Transformed data in reduced dimensions
    def fit_transform(data)
      validate_input(data)
      
      # Create RustUMAP with adjusted parameters if needed
      create_rust_umap_with_adjusted_params(data)
      
      result = Silence.maybe_silence do
        @rust_umap.fit_transform(data)
      end
      @fitted = true
      result
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
      rust_umap = ClusterKit::RustUMAP.load_model(path)
      
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
    
    def validate_input(data)
      raise ArgumentError, "Input must be an array" unless data.is_a?(Array)
      raise ArgumentError, "Input cannot be empty" if data.empty?
      
      first_row = data.first
      raise ArgumentError, "Input must be a 2D array (array of arrays)" unless first_row.is_a?(Array)
      
      row_length = first_row.length
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
          
          if val.nan? || val.infinite?
            raise ArgumentError, "Element at position [#{i}, #{j}] is NaN or Infinite"
          end
        end
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
        
        if ClusterKit.configuration.verbose
          warn "UMAP: Adjusted n_neighbors from #{@n_neighbors} to #{adjusted_n_neighbors} for dataset with #{n_samples} samples"
        end
      end
      
      @rust_umap = ClusterKit::RustUMAP.new({
        n_components: @n_components,
        n_neighbors: adjusted_n_neighbors,
        random_seed: @random_seed,
        nb_grad_batch: @nb_grad_batch,
        nb_sampling_by_edge: @nb_sampling_by_edge
      })
    end
  end
end