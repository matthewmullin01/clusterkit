# frozen_string_literal: true

require 'fileutils'

module AnnEmbed
  class UMAP
    def initialize(n_components: 2, n_neighbors: 15, random_seed: nil)
      @n_components = n_components
      @n_neighbors = n_neighbors
      @rust_umap = RustUMAP.new(
        n_components: n_components,
        n_neighbors: n_neighbors,
        random_seed: random_seed
      )
      @last_embeddings = nil
      @last_original_data = nil
    end
    
    # Load embeddings from a file
    def self.load_embeddings(path)
      raise ArgumentError, "File not found: #{path}" unless File.exist?(path)
      RustUMAP.load_embeddings(path)
    end
    
    # Save embeddings to a file 
    def self.save_embeddings(path, embeddings, original_data, options = {})
      # Ensure directory exists
      dir = File.dirname(path)
      FileUtils.mkdir_p(dir) unless dir == '.' || dir == '/'
      
      RustUMAP.save_embeddings(path, embeddings, original_data, options)
    end
    
    # Fit the model and transform the data in one step
    def fit_transform(data)
      validate_input(data)
      embeddings = @rust_umap.fit_transform(data)
      
      # Store for potential saving
      @last_embeddings = embeddings
      @last_original_data = data
      
      embeddings
    end
    
    # Save the last computed embeddings to a file
    def save_embeddings(path)
      raise RuntimeError, "No embeddings to save. Run fit_transform first." unless @last_embeddings
      
      self.class.save_embeddings(path, @last_embeddings, @last_original_data, {
        n_components: @n_components,
        n_neighbors: @n_neighbors
      })
    end
    
    # Save the trained UMAP model
    def save(path)
      raise RuntimeError, "No model to save. Run fit_transform first." unless @last_original_data
      
      # Ensure directory exists
      dir = File.dirname(path)
      FileUtils.mkdir_p(dir) unless dir == '.' || dir == '/'
      
      @rust_umap.save_model(path)
    end
    
    # Load a trained UMAP model
    def self.load(path)
      raise ArgumentError, "File not found: #{path}" unless File.exist?(path)
      
      # Load the Rust model
      rust_umap = RustUMAP.load_model(path)
      
      # Create a new UMAP instance with the loaded model
      instance = allocate
      instance.instance_variable_set(:@rust_umap, rust_umap)
      # We don't know the original parameters, but the model should work
      instance.instance_variable_set(:@n_components, nil)
      instance.instance_variable_set(:@n_neighbors, nil)
      instance.instance_variable_set(:@last_embeddings, nil)
      instance.instance_variable_set(:@last_original_data, nil)
      
      instance
    end
    
    # Transform new data using the fitted model
    def transform(data)
      validate_input(data)
      @rust_umap.transform(data)
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
  end
end