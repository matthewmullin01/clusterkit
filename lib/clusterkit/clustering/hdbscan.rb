# frozen_string_literal: true

require_relative '../data_validator'

module ClusterKit
  module Clustering
    # HDBSCAN clustering algorithm - matching KMeans API pattern
    class HDBSCAN
      attr_reader :min_samples, :min_cluster_size, :metric, :labels, :probabilities, 
                  :outlier_scores, :cluster_persistence

      # Initialize HDBSCAN clusterer (matches KMeans pattern)
      # @param min_samples [Integer] Min neighborhood size for core points (default: 5)
      # @param min_cluster_size [Integer] Minimum size of clusters (default: 5)
      # @param metric [String] Distance metric (default: 'euclidean')
      def initialize(min_samples: 5, min_cluster_size: 5, metric: 'euclidean')
        raise ArgumentError, "min_samples must be positive" unless min_samples > 0
        raise ArgumentError, "min_cluster_size must be positive" unless min_cluster_size > 0
        
        valid_metrics = ['euclidean', 'l2', 'manhattan', 'l1', 'cosine']
        unless valid_metrics.include?(metric)
          raise ArgumentError, "metric must be one of: #{valid_metrics.join(', ')}"
        end
        
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
        
        @labels = result["labels"]
        @probabilities = result["probabilities"]
        @outlier_scores = result["outlier_scores"]
        @cluster_persistence = result["cluster_persistence"]
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
        # Count unique labels excluding -1 (noise)
        unique_labels = @labels.uniq.reject { |l| l == -1 }
        unique_labels.length
      end

      # Get noise ratio (HDBSCAN-specific but follows naming pattern)
      # @return [Float] Fraction of points labeled as noise
      def noise_ratio
        return 0.0 unless fitted?
        @labels.count(-1).to_f / @labels.length
      end

      # Get the number of noise points
      # @return [Integer] Number of points labeled as noise
      def n_noise_points
        return 0 unless fitted?
        @labels.count(-1)
      end

      # Get indices of noise points
      # @return [Array<Integer>] Indices of points labeled as noise
      def noise_indices
        return [] unless fitted?
        @labels.each_with_index.select { |label, _| label == -1 }.map { |_, idx| idx }
      end

      # Get indices of points in each cluster
      # @return [Hash<Integer, Array<Integer>>] Mapping of cluster label to point indices
      def cluster_indices
        return {} unless fitted?
        
        result = {}
        @labels.each_with_index do |label, idx|
          next if label == -1  # Skip noise points
          result[label] ||= []
          result[label] << idx
        end
        result
      end

      # Get summary statistics
      # @return [Hash] Summary of clustering results
      def summary
        return {} unless fitted?
        
        {
          n_clusters: n_clusters,
          n_noise_points: n_noise_points,
          noise_ratio: noise_ratio,
          cluster_sizes: cluster_indices.transform_values(&:length),
          cluster_persistence: @cluster_persistence
        }
      end

      private

      def validate_data(data)
        # Use same validation as KMeans for consistency
        DataValidator.validate_clustering(data, check_finite: false)
      end
    end

    # Module-level convenience methods (matching KMeans pattern)
    class << self
      # Perform HDBSCAN clustering (matches Clustering.kmeans signature pattern)
      # @param data [Array] 2D array of data points
      # @param min_samples [Integer] Min neighborhood size for core points
      # @param min_cluster_size [Integer] Minimum size of clusters
      # @param metric [String] Distance metric
      # @return [Hash] Result hash with :labels, :probabilities, :outlier_scores
      def hdbscan(data, min_samples: 5, min_cluster_size: 5, metric: 'euclidean')
        clusterer = HDBSCAN.new(
          min_samples: min_samples,
          min_cluster_size: min_cluster_size,
          metric: metric
        )
        clusterer.fit(data)
        {
          labels: clusterer.labels,
          probabilities: clusterer.probabilities,
          outlier_scores: clusterer.outlier_scores,
          n_clusters: clusterer.n_clusters,
          noise_ratio: clusterer.noise_ratio,
          cluster_persistence: clusterer.cluster_persistence || {}
        }
      end
    end
  end
end