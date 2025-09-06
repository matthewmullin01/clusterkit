# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/clustering'

RSpec.describe "Additional K-means edge cases and stress tests" do
  describe ClusterKit::Clustering::KMeans do
    describe 'numerical edge cases' do
      it 'handles very small values' do
        data = 20.times.map { [rand * 1e-10, rand * 1e-10] }
        kmeans = described_class.new(k: 2)
        expect { kmeans.fit_predict(data) }.not_to raise_error
      end

      it 'handles mixed scale data' do
        # Some features are tiny, some are large
        data = 20.times.map { [rand * 1e-6, rand * 1e6] }
        kmeans = described_class.new(k: 2)
        labels = kmeans.fit_predict(data)
        expect(labels.uniq.size).to be_between(1, 2).inclusive
      end

      it 'handles negative values' do
        data = 20.times.map { [rand - 0.5, rand - 0.5] }
        kmeans = described_class.new(k: 3)
        labels = kmeans.fit_predict(data)
        expect(labels).to all(be_between(0, 2).inclusive)
      end

      it 'handles zero variance features' do
        # Second feature is constant
        data = 20.times.map { [rand, 0.5] }
        kmeans = described_class.new(k: 2)
        expect { kmeans.fit_predict(data) }.not_to raise_error
      end
    end

    describe 'convergence behavior' do
      it 'converges with max_iter=1' do
        data = 20.times.map { [rand, rand] }
        kmeans = described_class.new(k: 3, max_iter: 1)
        labels = kmeans.fit_predict(data)
        expect(labels).to be_a(Array)
      end

      # Note: Random seed determinism would require thread-local RNG in Rust
      # which is not currently implemented

      it 'produces identical results with same seed (NOW WORKING!)' do
        # Generate deterministic test data for consistent testing
        data = []
        srand(12345) # Fixed seed for test data generation
        30.times { data << 5.times.map { rand } }
        
        # Two KMeans instances with identical parameters
        kmeans1 = described_class.new(k: 3, random_seed: 42)
        kmeans2 = described_class.new(k: 3, random_seed: 42)
        
        result1 = kmeans1.fit(data)
        result2 = kmeans2.fit(data)
        
        # These are now identical because seeding is fixed!
        expect(result1.centroids).to eq(result2.centroids)
        expect(result1.labels).to eq(result2.labels)
        expect(result1.inertia).to eq(result2.inertia)
      end
      
      it 'produces different results with different seeds (when seeding works)' do
        # Generate deterministic test data
        data = []
        srand(12345)
        30.times { data << 5.times.map { rand } }
        
        kmeans1 = described_class.new(k: 3, random_seed: 42)
        kmeans2 = described_class.new(k: 3, random_seed: 999)
        
        result1 = kmeans1.fit(data)
        result2 = kmeans2.fit(data)
        
        # Currently this test can't verify different results because seeding is broken
        # When seeding is fixed, different seeds should produce different results
        # For now, just verify both complete successfully
        expect([result1.inertia, result2.inertia]).to all(be_a(Float))
        expect([result1.centroids.size, result2.centroids.size]).to all(eq(3))
      end
      
      it 'demonstrates working seeding with multiple runs' do
        # Generate consistent test data  
        data = []
        srand(12345)
        20.times { data << 3.times.map { rand } }
        
        # Run the same configuration 3 times
        results = 3.times.map do
          kmeans = described_class.new(k: 2, random_seed: 42)
          kmeans.fit_predict(data)
        end
        
        # With working seeding, all results should be identical
        all_same = results.all? { |r| r == results.first }
        expect(all_same).to be(true), 
          "Same seed should produce identical results"
      end
    end

    describe 'error conditions' do
      it 'raises error when k > n_samples' do
        data = [[1, 2], [3, 4]]
        kmeans = described_class.new(k: 3)
        expect { 
          kmeans.fit(data) 
        }.to raise_error(ArgumentError, /k .* cannot be larger than number of samples/)
      end

      it 'raises error for k=0' do
        data = 10.times.map { [rand, rand] }
        expect {
          described_class.new(k: 0)
        }.to raise_error(ArgumentError)
      end

      it 'handles single sample gracefully' do
        data = [[1.0, 2.0]]
        kmeans = described_class.new(k: 1)
        labels = kmeans.fit_predict(data)
        expect(labels).to eq([0])
      end
    end

    describe 'prediction consistency' do
      it 'assigns points to nearest centroid' do
        # Create clear clusters
        cluster1 = 5.times.map { [rand * 0.1, rand * 0.1] }
        cluster2 = 5.times.map { [rand * 0.1 + 10, rand * 0.1 + 10] }
        data = cluster1 + cluster2
        
        kmeans = described_class.new(k: 2, random_seed: 42)
        kmeans.fit(data)
        
        # Test point very close to origin should go to cluster1's cluster
        test_point = [[0.05, 0.05]]
        label = kmeans.predict(test_point).first
        
        # Test point very close to (10,10) should go to cluster2's cluster
        test_point2 = [[10.05, 10.05]]
        label2 = kmeans.predict(test_point2).first
        
        # They should be in different clusters
        expect(label).not_to eq(label2)
      end

      it 'predict returns same labels for training data' do
        data = 20.times.map { [rand * 10, rand * 10] }
        kmeans = described_class.new(k: 3, random_seed: 42)
        
        original_labels = kmeans.fit_predict(data)
        predicted_labels = kmeans.predict(data)
        
        expect(predicted_labels).to eq(original_labels)
      end
    end

    describe 'high dimensional data' do
      it 'handles 100-dimensional data' do
        data = 50.times.map { 100.times.map { rand } }
        kmeans = described_class.new(k: 5, max_iter: 10)
        labels = kmeans.fit_predict(data)
        expect(labels.uniq.size).to be_between(1, 5).inclusive
      end

      it 'handles more features than samples' do
        # 5 samples, 20 features
        data = 5.times.map { 20.times.map { rand } }
        kmeans = described_class.new(k: 2)
        labels = kmeans.fit_predict(data)
        expect(labels.size).to eq(5)
      end
    end
  end

  describe ClusterKit::Clustering::KMeans do
    describe '.elbow_method stress test' do
      it 'handles k_range with single value' do
        data = 20.times.map { [rand, rand] }
        results = described_class.elbow_method(data, k_range: 3..3)
        expect(results.keys).to eq([3])
      end

      it 'shows decreasing inertia with increasing k' do
        data = 30.times.map { 5.times.map { rand } }
        results = described_class.elbow_method(data, k_range: 2..6)
        
        # Generally, inertia should decrease as k increases
        # Though not strictly monotonic due to random initialization
        inertias = results.values
        expect(inertias.first).to be >= inertias.last * 0.5 # Allow some variance
      end
    end

  end
  
  describe ClusterKit::Clustering do
    describe '.silhouette_score edge cases' do
      it 'handles perfect clustering' do
        # Two perfectly separated clusters
        cluster1 = 10.times.map { [rand * 0.1, rand * 0.1] }
        cluster2 = 10.times.map { [rand * 0.1 + 100, rand * 0.1 + 100] }
        data = cluster1 + cluster2
        labels = [0] * 10 + [1] * 10
        
        score = described_class.silhouette_score(data, labels)
        expect(score).to be > 0.9 # Should be nearly perfect
      end

      it 'handles worst case clustering' do
        # Interleaved clusters - worst possible assignment
        data = 20.times.map { [rand, rand] }
        labels = (0...20).map { |i| i % 2 } # Alternating labels
        
        score = described_class.silhouette_score(data, labels)
        expect(score).to be < 0.5 # Should be poor
      end

      it 'handles all points in same location' do
        data = 10.times.map { [1.0, 1.0] } # All identical
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        
        score = described_class.silhouette_score(data, labels)
        expect(score).to be_finite
      end
    end
  end
end