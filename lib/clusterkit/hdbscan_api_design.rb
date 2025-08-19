# API Design for HDBSCAN to match KMeans pattern

module ClusterKit
  module Clustering
    
    # HDBSCAN clustering algorithm - matching KMeans API pattern
    class HDBSCAN
      attr_reader :min_samples, :min_cluster_size, :labels, :probabilities, 
                  :outlier_scores, :cluster_persistence

      # Initialize HDBSCAN clusterer (matches KMeans pattern)
      # @param min_samples [Integer] Min neighborhood size for core points (default: 5)
      # @param min_cluster_size [Integer] Minimum size of clusters (default: 5)
      # @param metric [String] Distance metric (default: 'euclidean')
      def initialize(min_samples: 5, min_cluster_size: 5, metric: 'euclidean')
        raise ArgumentError, "min_samples must be positive" unless min_samples > 0
        raise ArgumentError, "min_cluster_size must be positive" unless min_cluster_size > 0
        @min_samples = min_samples
        @min_cluster_size = min_cluster_size
        @metric = metric
        @fitted = false
      end

      # Fit the HDBSCAN model (matches KMeans.fit)
      # @param data [Array] 2D array of data points
      # @return [self] Returns self for method chaining
      def fit(data)
        validate_data(data)
        
        # Call Rust implementation (hdbscan crate)
        result = Clustering.hdbscan_rust(data, @min_samples, @min_cluster_size, @metric)
        
        @labels = result[:labels]
        @probabilities = result[:probabilities]
        @outlier_scores = result[:outlier_scores]
        @cluster_persistence = result[:cluster_persistence]
        @fitted = true
        
        self
      end

      # HDBSCAN doesn't support predict for new points (unlike KMeans)
      # But we keep the method for API consistency
      # @param data [Array] 2D array of data points
      # @return [Array] Returns nil or raises
      def predict(data)
        raise NotImplementedError, "HDBSCAN does not support prediction on new data. " \
                                  "Use approximate_predict for approximate membership"
      end

      # Fit the model and return labels (matches KMeans.fit_predict)
      # @param data [Array] 2D array of data points
      # @return [Array] Cluster labels (-1 for noise)
      def fit_predict(data)
        fit(data)
        @labels
      end

      # Check if model has been fitted (matches KMeans.fitted?)
      # @return [Boolean] True if fitted
      def fitted?
        @fitted
      end

      # Get number of clusters found (similar to KMeans.k but discovered)
      # @return [Integer] Number of clusters (excluding noise)
      def n_clusters
        return 0 unless fitted?
        @labels.max + 1 rescue 0
      end

      # Get noise ratio (HDBSCAN-specific but follows naming pattern)
      # @return [Float] Fraction of points labeled as noise
      def noise_ratio
        return 0.0 unless fitted?
        @labels.count(-1).to_f / @labels.length
      end

      private

      def validate_data(data)
        # Exact same validation as KMeans for consistency
        raise ArgumentError, "Data must be an array" unless data.is_a?(Array)
        raise ArgumentError, "Data cannot be empty" if data.empty?
        raise ArgumentError, "Data must be 2D array" unless data.first.is_a?(Array)
        
        row_length = data.first.length
        unless data.all? { |row| row.is_a?(Array) && row.length == row_length }
          raise ArgumentError, "All rows must have the same length"
        end
        
        data.each_with_index do |row, i|
          row.each_with_index do |val, j|
            unless val.is_a?(Numeric)
              raise ArgumentError, "Element at position [#{i}, #{j}] is not numeric"
            end
          end
        end
      end
    end

    # Module-level convenience methods (matching KMeans pattern)
    class << self
      # Perform HDBSCAN clustering (matches Clustering.kmeans signature)
      # @param data [Array] 2D array of data points
      # @param min_samples [Integer] Min neighborhood size for core points
      # @param min_cluster_size [Integer] Minimum size of clusters
      # @return [Hash] Result hash with :labels, :probabilities, :outlier_scores
      def hdbscan(data, min_samples: 5, min_cluster_size: 5)
        clusterer = HDBSCAN.new(min_samples: min_samples, min_cluster_size: min_cluster_size)
        clusterer.fit(data)
        {
          labels: clusterer.labels,
          probabilities: clusterer.probabilities,
          outlier_scores: clusterer.outlier_scores,
          n_clusters: clusterer.n_clusters,
          noise_ratio: clusterer.noise_ratio
        }
      end
    end
  end
end

# Usage comparison:

# KMeans usage:
kmeans = ClusterKit::Clustering::KMeans.new(k: 3)
kmeans.fit(data)
labels = kmeans.labels
# or
labels = kmeans.fit_predict(data)

# HDBSCAN usage (identical pattern):
hdbscan = ClusterKit::Clustering::HDBSCAN.new(min_samples: 5, min_cluster_size: 5)
hdbscan.fit(data)
labels = hdbscan.labels
# or
labels = hdbscan.fit_predict(data)

# Module-level convenience (both follow same pattern):
result = ClusterKit::Clustering.kmeans(data, 3)
result = ClusterKit::Clustering.hdbscan(data, min_samples: 5)