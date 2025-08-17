# frozen_string_literal: true

require_relative 'configuration'
require_relative 'silence'

module ClusterKit
  # Main class for performing dimensionality reduction
  class Embedder
    attr_reader :method, :n_components, :config

    # Initialize a new embedder
    # @param method [Symbol] The embedding method (:umap, :tsne, :largevis, :diffusion)
    # @param n_components [Integer] Number of output dimensions
    # @param options [Hash] Additional configuration options
    def initialize(method: :umap, n_components: 2, **options)
      unless METHODS.include?(method)
        raise ArgumentError, "Unknown method: #{method}. Must be one of: #{METHODS.join(', ')}"
      end

      @method = method
      @n_components = n_components
      @config = Config.new(method: method, n_components: n_components, **options)
      @rust_embedder = nil
      @fitted = false
    end

    # Fit the embedder and transform data in one step
    # @param data [Array, Numo::NArray] Input data
    # @return [Numo::NArray] Embedded data
    def fit_transform(data)
      data_array = prepare_data(data)
      
      # Automatically adjust n_neighbors if it's too high for the dataset
      adjust_n_neighbors_for_data_size(data_array)
      
      result = Silence.maybe_silence do
        @rust_embedder = ClusterKit::RustUMAP.new(@config.to_h)
        @rust_embedder.fit_transform(data_array)
      end
      @fitted = true
      
      result
    end

    # Fit the embedder to data
    # @param data [Array, Numo::NArray] Input data
    # @return [self]
    def fit(data)
      data_array = prepare_data(data)
      
      # Automatically adjust n_neighbors if it's too high for the dataset
      adjust_n_neighbors_for_data_size(data_array)
      
      Silence.maybe_silence do
        @rust_embedder = ClusterKit::RustUMAP.new(@config.to_h)
        # RustUMAP doesn't have a separate fit method, so we use fit_transform and discard the result
        @rust_embedder.fit_transform(data_array)
      end
      @fitted = true
      
      self
    end

    # Transform new data using fitted embedder
    # @param data [Array, Numo::NArray] Input data
    # @return [Numo::NArray] Embedded data
    def transform(data)
      raise Error, "Embedder must be fitted before transform" unless fitted?
      
      data_array = prepare_data(data)
      Silence.maybe_silence do
        @rust_embedder.transform(data_array)
      end
    end

    # Check if embedder has been fitted
    # @return [Boolean]
    def fitted?
      @fitted
    end

    # Save the embedder to a file
    # @param path [String] File path
    def save(path)
      raise Error, "Embedder must be fitted before saving" unless fitted?
      
      @rust_embedder.save_model(path)
    end

    # Load an embedder from a file
    # @param path [String] File path
    # @return [Embedder] Loaded embedder
    def self.load(path)
      rust_embedder = ClusterKit::RustUMAP.load_model(path)
      embedder = allocate
      embedder.instance_variable_set(:@rust_embedder, rust_embedder)
      embedder.instance_variable_set(:@fitted, true)
      # TODO: Restore config from saved model
      embedder
    end

    private

    def prepare_data(data)
      case data
      when Array
        # Keep as array for RustUMAP
        data
      when String
        # Assume it's a file path
        load_csv_data(data)
      else
        raise ArgumentError, "Unsupported data type: #{data.class}. Expected Array or String (CSV path)"
      end
    end

    def load_csv_data(path)
      require "csv"
      CSV.read(path, converters: :numeric)
    end
    
    def adjust_n_neighbors_for_data_size(data_array)
      n_samples = data_array.size
      
      # Only adjust if using UMAP-like methods that use n_neighbors
      return unless [:umap, :largevis].include?(@method)
      
      # n_neighbors should be less than n_samples
      # Use a reasonable default: min(15, n_samples / 4) but at least 2
      max_neighbors = [n_samples - 1, 2].max  # At least 2, but less than n_samples
      suggested_neighbors = [[15, n_samples / 4].min.to_i, 2].max
      
      if @config.n_neighbors > max_neighbors
        old_value = @config.n_neighbors
        @config.n_neighbors = [suggested_neighbors, max_neighbors].min
        
        if ClusterKit.configuration.verbose
          warn "Adjusted n_neighbors from #{old_value} to #{@config.n_neighbors} for dataset with #{n_samples} samples"
        end
      end
    end
  end
end