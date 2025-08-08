# frozen_string_literal: true

module AnnEmbed
  class UMAP
    def initialize(n_components: 2, n_neighbors: 15, random_seed: nil)
      @rust_umap = RustUMAP.new(
        n_components: n_components,
        n_neighbors: n_neighbors,
        random_seed: random_seed
      )
    end
    
    def fit_transform(data)
      validate_input(data)
      @rust_umap.fit_transform(data)
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