# frozen_string_literal: true

module ClusterKit
  # Shared data validation methods for all algorithms
  module DataValidator
    class << self
      # Validate basic data structure and types
      # @param data [Array] Data to validate
      # @raise [ArgumentError] If data structure is invalid
      def validate_basic_structure(data)
        raise ArgumentError, "Input must be an array" unless data.is_a?(Array)
        raise ArgumentError, "Input cannot be empty" if data.empty?

        first_row = data.first
        raise ArgumentError, "Input must be a 2D array (array of arrays)" unless first_row.is_a?(Array)
      end

      # Validate row consistency (all rows have same length)
      # @param data [Array] 2D array to validate
      # @raise [ArgumentError] If rows have different lengths
      def validate_row_consistency(data)
        row_length = data.first.length

        data.each_with_index do |row, i|
          unless row.is_a?(Array)
            raise ArgumentError, "Row #{i} is not an array"
          end

          if row.length != row_length
            raise ArgumentError, "All rows must have the same length (row #{i} has #{row.length} elements, expected #{row_length})"
          end
        end
      end

      # Validate that all elements are numeric
      # @param data [Array] 2D array to validate
      # @raise [ArgumentError] If any element is not numeric
      def validate_numeric_types(data)
        data.each_with_index do |row, i|
          row.each_with_index do |val, j|
            unless val.is_a?(Numeric)
              raise ArgumentError, "Element at position [#{i}, #{j}] is not numeric"
            end
          end
        end
      end

      # Validate finite values (no NaN or Infinite)
      # @param data [Array] 2D array to validate
      # @raise [ArgumentError] If any float is NaN or Infinite
      def validate_finite_values(data)
        data.each_with_index do |row, i|
          row.each_with_index do |val, j|
            # Only check for NaN/Infinite on floats
            if val.is_a?(Float) && (val.nan? || val.infinite?)
              raise ArgumentError, "Element at position [#{i}, #{j}] is NaN or Infinite"
            end
          end
        end
      end

      # Standard validation for most algorithms
      # @param data [Array] 2D array to validate
      # @param check_finite [Boolean] Whether to check for NaN/Infinite values
      # @raise [ArgumentError] If data is invalid
      def validate_standard(data, check_finite: true)
        validate_basic_structure(data)
        validate_row_consistency(data)
        validate_numeric_types(data)
        validate_finite_values(data) if check_finite
      end

      # Validation for clustering algorithms (KMeans, HDBSCAN) with specific error messages
      # @param data [Array] 2D array to validate
      # @param check_finite [Boolean] Whether to check for NaN/Infinite values
      # @raise [ArgumentError] If data is invalid
      def validate_clustering(data, check_finite: false)
        raise ArgumentError, "Data must be an array" unless data.is_a?(Array)
        raise ArgumentError, "Data cannot be empty" if data.empty?
        raise ArgumentError, "Data must be 2D array" unless data.first.is_a?(Array)

        validate_row_consistency(data)
        validate_numeric_types(data)
        validate_finite_values(data) if check_finite
      end

      # Validation for PCA with specific error messages (same as clustering but without finite checks)
      # @param data [Array] 2D array to validate
      # @raise [ArgumentError] If data is invalid
      def validate_pca(data)
        raise ArgumentError, "Data must be an array" unless data.is_a?(Array)
        raise ArgumentError, "Data cannot be empty" if data.empty?
        raise ArgumentError, "Data must be 2D array" unless data.first.is_a?(Array)

        validate_row_consistency(data)
        validate_numeric_types(data)
      end

      # Get data statistics for warnings/error context
      # @param data [Array] 2D array
      # @return [Hash] Statistics about the data
      def data_statistics(data)
        return { n_samples: 0, n_features: 0, data_range: 0.0 } if data.empty?

        n_samples = data.size
        n_features = data.first&.size || 0
        
        # Calculate data range for warnings
        min_val = Float::INFINITY
        max_val = -Float::INFINITY

        data.each do |row|
          row.each do |val|
            val_f = val.to_f
            min_val = val_f if val_f < min_val
            max_val = val_f if val_f > max_val
          end
        end

        data_range = max_val - min_val

        {
          n_samples: n_samples,
          n_features: n_features,
          data_range: data_range,
          min_value: min_val,
          max_value: max_val
        }
      end
    end
  end
end