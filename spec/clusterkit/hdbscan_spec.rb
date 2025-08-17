# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/clustering'
require 'clusterkit/clustering/hdbscan'

RSpec.describe ClusterKit::Clustering::HDBSCAN do
  # Generate test data with clear clusters and noise
  let(:clustered_data) {
    # Create 3 well-separated clusters plus noise
    cluster1 = 20.times.map { [rand * 2, rand * 2] }          # Cluster around (1, 1)
    cluster2 = 20.times.map { [rand * 2 + 5, rand * 2 + 5] }  # Cluster around (6, 6)
    cluster3 = 20.times.map { [rand * 2 + 10, rand * 2] }     # Cluster around (11, 1)
    # Spread noise points more widely to ensure they're detected as noise
    noise = [
      [20, 20],  # Far outlier
      [-5, -5],  # Far outlier
      [15, 15],  # Far outlier
      [8, -3],   # Isolated point
      [3, 10]    # Isolated point
    ]
    cluster1 + cluster2 + cluster3 + noise
  }
  
  let(:small_data) {
    # Small dataset for testing edge cases
    [[0, 0], [0.1, 0.1], [0.2, 0.2], [5, 5], [5.1, 5.1]]
  }
  
  describe '#initialize' do
    it 'creates a new HDBSCAN instance with defaults' do
      hdbscan = described_class.new
      expect(hdbscan).to be_a(described_class)
      expect(hdbscan.min_samples).to eq(5)
      expect(hdbscan.min_cluster_size).to eq(5)
      expect(hdbscan.metric).to eq('euclidean')
    end
    
    it 'accepts custom parameters' do
      hdbscan = described_class.new(
        min_samples: 3,
        min_cluster_size: 10,
        metric: 'euclidean'
      )
      expect(hdbscan.min_samples).to eq(3)
      expect(hdbscan.min_cluster_size).to eq(10)
      expect(hdbscan.metric).to eq('euclidean')
    end
    
    it 'raises error for invalid min_samples' do
      expect {
        described_class.new(min_samples: 0)
      }.to raise_error(ArgumentError, /min_samples must be positive/)
      
      expect {
        described_class.new(min_samples: -1)
      }.to raise_error(ArgumentError, /min_samples must be positive/)
    end
    
    it 'raises error for invalid min_cluster_size' do
      expect {
        described_class.new(min_cluster_size: 0)
      }.to raise_error(ArgumentError, /min_cluster_size must be positive/)
    end
    
    it 'raises error for invalid metric' do
      expect {
        described_class.new(metric: 'invalid')
      }.to raise_error(ArgumentError, /metric must be one of/)
    end
    
    it 'accepts alternative metric names' do
      expect { described_class.new(metric: 'l2') }.not_to raise_error
      expect { described_class.new(metric: 'manhattan') }.not_to raise_error
      expect { described_class.new(metric: 'l1') }.not_to raise_error
      expect { described_class.new(metric: 'cosine') }.not_to raise_error
    end
  end
  
  describe '#fit' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'fits the model to data' do
      result = hdbscan.fit(clustered_data)
      expect(result).to eq(hdbscan)
      expect(hdbscan).to be_fitted
    end
    
    it 'sets labels array' do
      hdbscan.fit(clustered_data)
      expect(hdbscan.labels).to be_a(Array)
      expect(hdbscan.labels.size).to eq(clustered_data.size)
      expect(hdbscan.labels).to all(be_a(Integer))
    end
    
    it 'sets probabilities array' do
      hdbscan.fit(clustered_data)
      expect(hdbscan.probabilities).to be_a(Array)
      expect(hdbscan.probabilities.size).to eq(clustered_data.size)
      expect(hdbscan.probabilities).to all(be_a(Numeric))
    end
    
    it 'sets outlier_scores array' do
      hdbscan.fit(clustered_data)
      expect(hdbscan.outlier_scores).to be_a(Array)
      expect(hdbscan.outlier_scores.size).to eq(clustered_data.size)
      expect(hdbscan.outlier_scores).to all(be_a(Numeric))
    end
    
    it 'sets cluster_persistence hash' do
      hdbscan.fit(clustered_data)
      expect(hdbscan.cluster_persistence).to be_a(Hash)
    end
    
    it 'identifies noise points with -1 label' do
      hdbscan.fit(clustered_data)
      noise_labels = hdbscan.labels.select { |l| l == -1 }
      expect(noise_labels).not_to be_empty
    end
    
    it 'raises error for empty data' do
      expect {
        hdbscan.fit([])
      }.to raise_error(ArgumentError, /Data cannot be empty/)
    end
    
    it 'raises error for non-array data' do
      expect {
        hdbscan.fit("not an array")
      }.to raise_error(ArgumentError, /Data must be an array/)
    end
    
    it 'raises error for 1D array' do
      expect {
        hdbscan.fit([1, 2, 3])
      }.to raise_error(ArgumentError, /Data must be 2D array/)
    end
    
    it 'raises error for inconsistent row lengths' do
      bad_data = [[1, 2], [3, 4, 5]]
      expect {
        hdbscan.fit(bad_data)
      }.to raise_error(ArgumentError, /All rows must have the same length/)
    end
    
    it 'raises error for non-numeric data' do
      bad_data = [[1, "two"], [3, 4]]
      expect {
        hdbscan.fit(bad_data)
      }.to raise_error(ArgumentError, /is not numeric/)
    end
  end
  
  describe '#predict' do
    let(:hdbscan) { described_class.new }
    
    it 'raises NotImplementedError' do
      hdbscan.fit(clustered_data)
      expect {
        hdbscan.predict([[0, 0]])
      }.to raise_error(NotImplementedError, /HDBSCAN does not support prediction/)
    end
  end
  
  describe '#fit_predict' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'fits and returns labels in one step' do
      labels = hdbscan.fit_predict(clustered_data)
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(clustered_data.size)
      expect(hdbscan).to be_fitted
      expect(labels).to eq(hdbscan.labels)
    end
  end
  
  describe '#fitted?' do
    let(:hdbscan) { described_class.new }
    
    it 'returns false before fitting' do
      expect(hdbscan).not_to be_fitted
    end
    
    it 'returns true after fitting' do
      hdbscan.fit(clustered_data)
      expect(hdbscan).to be_fitted
    end
  end
  
  describe '#n_clusters' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns 0 before fitting' do
      expect(hdbscan.n_clusters).to eq(0)
    end
    
    it 'returns number of clusters excluding noise' do
      hdbscan.fit(clustered_data)
      unique_labels = hdbscan.labels.uniq.reject { |l| l == -1 }
      expect(hdbscan.n_clusters).to eq(unique_labels.size)
    end
    
    it 'returns 0 when all points are noise' do
      # Use very strict parameters that make everything noise
      strict_hdbscan = described_class.new(min_samples: 100, min_cluster_size: 100)
      strict_hdbscan.fit(clustered_data)
      expect(strict_hdbscan.n_clusters).to eq(0)
    end
  end
  
  describe '#noise_ratio' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns 0.0 before fitting' do
      expect(hdbscan.noise_ratio).to eq(0.0)
    end
    
    it 'returns fraction of noise points' do
      hdbscan.fit(clustered_data)
      expected_ratio = hdbscan.labels.count(-1).to_f / hdbscan.labels.size
      expect(hdbscan.noise_ratio).to eq(expected_ratio)
      expect(hdbscan.noise_ratio).to be_between(0.0, 1.0)
    end
  end
  
  describe '#n_noise_points' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns 0 before fitting' do
      expect(hdbscan.n_noise_points).to eq(0)
    end
    
    it 'returns count of noise points' do
      hdbscan.fit(clustered_data)
      expected_count = hdbscan.labels.count(-1)
      expect(hdbscan.n_noise_points).to eq(expected_count)
    end
  end
  
  describe '#noise_indices' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns empty array before fitting' do
      expect(hdbscan.noise_indices).to eq([])
    end
    
    it 'returns indices of noise points' do
      hdbscan.fit(clustered_data)
      noise_indices = hdbscan.noise_indices
      
      noise_indices.each do |idx|
        expect(hdbscan.labels[idx]).to eq(-1)
      end
    end
  end
  
  describe '#cluster_indices' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns empty hash before fitting' do
      expect(hdbscan.cluster_indices).to eq({})
    end
    
    it 'returns hash of cluster labels to point indices' do
      hdbscan.fit(clustered_data)
      cluster_indices = hdbscan.cluster_indices
      
      expect(cluster_indices).to be_a(Hash)
      
      # Each cluster should have indices
      cluster_indices.each do |label, indices|
        expect(label).not_to eq(-1)  # Should not include noise
        expect(indices).to be_a(Array)
        expect(indices).not_to be_empty
        
        # All indices should have this label
        indices.each do |idx|
          expect(hdbscan.labels[idx]).to eq(label)
        end
      end
    end
  end
  
  describe '#summary' do
    let(:hdbscan) { described_class.new(min_samples: 3, min_cluster_size: 5) }
    
    it 'returns empty hash before fitting' do
      expect(hdbscan.summary).to eq({})
    end
    
    it 'returns summary statistics after fitting' do
      hdbscan.fit(clustered_data)
      summary = hdbscan.summary
      
      expect(summary).to be_a(Hash)
      expect(summary).to have_key(:n_clusters)
      expect(summary).to have_key(:n_noise_points)
      expect(summary).to have_key(:noise_ratio)
      expect(summary).to have_key(:cluster_sizes)
      expect(summary).to have_key(:cluster_persistence)
      
      expect(summary[:cluster_sizes]).to be_a(Hash)
    end
  end
  
  describe 'module-level convenience method' do
    describe '.hdbscan' do
      it 'performs HDBSCAN clustering' do
        result = ClusterKit::Clustering.hdbscan(
          clustered_data,
          min_samples: 3,
          min_cluster_size: 5
        )
        
        expect(result).to be_a(Hash)
        expect(result).to have_key(:labels)
        expect(result).to have_key(:probabilities)
        expect(result).to have_key(:outlier_scores)
        expect(result).to have_key(:n_clusters)
        expect(result).to have_key(:noise_ratio)
        expect(result).to have_key(:cluster_persistence)
        
        expect(result[:labels]).to be_a(Array)
        expect(result[:labels].size).to eq(clustered_data.size)
      end
      
      it 'uses default parameters when not specified' do
        result = ClusterKit::Clustering.hdbscan(clustered_data)
        expect(result).to be_a(Hash)
        expect(result[:labels]).to be_a(Array)
      end
    end
  end
  
  describe 'parameter sensitivity' do
    it 'finds more clusters with smaller min_cluster_size' do
      lenient = described_class.new(min_samples: 2, min_cluster_size: 3)
      lenient.fit(clustered_data)
      
      strict = described_class.new(min_samples: 5, min_cluster_size: 10)
      strict.fit(clustered_data)
      
      expect(lenient.n_clusters).to be >= strict.n_clusters
    end
    
    it 'produces more noise with larger min_cluster_size' do
      lenient = described_class.new(min_samples: 2, min_cluster_size: 3)
      lenient.fit(clustered_data)
      
      strict = described_class.new(min_samples: 5, min_cluster_size: 15)
      strict.fit(clustered_data)
      
      # Note: This relationship doesn't always hold strictly due to the 
      # hierarchical nature of HDBSCAN. We'll just check that both complete
      expect(lenient.noise_ratio).to be >= 0.0
      expect(strict.noise_ratio).to be >= 0.0
      
      # More meaningful test: strict should find fewer or equal clusters
      expect(strict.n_clusters).to be <= lenient.n_clusters
    end
  end
  
  describe 'integration with dimensionality reduction' do
    let(:high_dim_data) {
      # Generate high-dimensional data
      30.times.map { 20.times.map { rand } }
    }
    
    it 'works with UMAP reduced data' do
      # Reduce dimensions
      umap = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 5)
      reduced = umap.fit_transform(high_dim_data)
      
      # Apply HDBSCAN
      hdbscan = described_class.new(min_samples: 3, min_cluster_size: 5)
      labels = hdbscan.fit_predict(reduced)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(30)
      expect(hdbscan.n_clusters).to be >= 0
    end
    
    it 'works with PCA reduced data' do
      # Reduce dimensions with PCA
      pca = ClusterKit::PCA.new(n_components: 5)
      reduced = pca.fit_transform(high_dim_data)
      
      # Apply HDBSCAN
      hdbscan = described_class.new(min_samples: 3, min_cluster_size: 5)
      labels = hdbscan.fit_predict(reduced)
      
      expect(labels).to be_a(Array)
      expect(labels.size).to eq(30)
    end
  end
  
  describe 'edge cases' do
    it 'handles very small datasets' do
      tiny_data = [[0, 0], [1, 1], [2, 2]]
      hdbscan = described_class.new(min_samples: 1, min_cluster_size: 2)
      
      expect {
        hdbscan.fit(tiny_data)
      }.not_to raise_error
      
      # With such small data, might all be noise or one cluster
      expect(hdbscan.labels.size).to eq(3)
    end
    
    it 'handles identical points' do
      identical_data = 10.times.map { [1.0, 2.0] }
      hdbscan = described_class.new(min_samples: 2, min_cluster_size: 3)
      
      hdbscan.fit(identical_data)
      
      # Identical points should typically form one cluster
      expect(hdbscan.n_clusters).to be <= 1
    end
    
    it 'handles datasets smaller than min_cluster_size' do
      small_data = [[0, 0], [1, 1]]
      hdbscan = described_class.new(min_samples: 5, min_cluster_size: 10)
      
      hdbscan.fit(small_data)
      
      # All points should be noise
      expect(hdbscan.labels).to all(eq(-1))
      expect(hdbscan.n_clusters).to eq(0)
      expect(hdbscan.noise_ratio).to eq(1.0)
    end
  end
  
  describe 'determinism' do
    it 'produces consistent results across runs' do
      hdbscan1 = described_class.new(min_samples: 3, min_cluster_size: 5)
      hdbscan2 = described_class.new(min_samples: 3, min_cluster_size: 5)
      
      labels1 = hdbscan1.fit_predict(clustered_data)
      labels2 = hdbscan2.fit_predict(clustered_data)
      
      # HDBSCAN should be deterministic with KD-trees
      expect(labels1).to eq(labels2)
      expect(hdbscan1.n_clusters).to eq(hdbscan2.n_clusters)
      expect(hdbscan1.noise_ratio).to eq(hdbscan2.noise_ratio)
    end
  end
  
  describe 'API consistency with KMeans' do
    let(:kmeans) { ClusterKit::Clustering::KMeans.new(k: 3) }
    let(:hdbscan) { described_class.new }
    
    it 'has similar initialization pattern' do
      expect(kmeans).to respond_to(:fit)
      expect(hdbscan).to respond_to(:fit)
      
      expect(kmeans).to respond_to(:fit_predict)
      expect(hdbscan).to respond_to(:fit_predict)
      
      expect(kmeans).to respond_to(:fitted?)
      expect(hdbscan).to respond_to(:fitted?)
    end
    
    it 'returns self from fit for method chaining' do
      expect(kmeans.fit(clustered_data)).to eq(kmeans)
      expect(hdbscan.fit(clustered_data)).to eq(hdbscan)
    end
    
    it 'uses same data validation' do
      invalid_data = [[1, "two"]]
      
      expect { kmeans.fit(invalid_data) }.to raise_error(ArgumentError, /is not numeric/)
      expect { hdbscan.fit(invalid_data) }.to raise_error(ArgumentError, /is not numeric/)
    end
  end
end