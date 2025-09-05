# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/clustering'

RSpec.describe "KMeans Random Seeding" do
  let(:test_data) do
    # Generate deterministic test data for consistent testing
    srand(12345)
    30.times.map { 5.times.map { rand * 10 } }
  end

  describe "fixed seeding behavior" do
    it "demonstrates that random_seed parameter is now working" do
      # SEEDING IS NOW FIXED! Same seeds should produce identical results
      
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
      
      result1 = kmeans1.fit(test_data)
      result2 = kmeans2.fit(test_data)
      
      # Same seeds now produce identical results!
      expect(result1.centroids).to eq(result2.centroids)
      expect(result1.labels).to eq(result2.labels)
      expect(result1.inertia).to eq(result2.inertia)
    end
    
    it "shows that Rust implementation now uses the passed seed" do
      # The seed is now properly passed to Rust and used
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: 999)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: 999)
      
      labels1 = kmeans1.fit_predict(test_data)
      labels2 = kmeans2.fit_predict(test_data)
      
      # Should be identical with working seeding
      expect(labels1).to eq(labels2)
    end
    
    it "demonstrates deterministic behavior across multiple runs" do
      # Multiple runs with same seed should be identical
      results = 5.times.map do
        kmeans = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
        kmeans.fit(test_data)
        {
          centroids: kmeans.centroids,
          labels: kmeans.labels,
          inertia: kmeans.inertia
        }
      end
      
      # All results should be identical with working seeding
      first_result = results.first
      all_identical = results.all? do |result|
        result[:centroids] == first_result[:centroids] &&
        result[:labels] == first_result[:labels] &&
        result[:inertia] == first_result[:inertia]
      end
      
      expect(all_identical).to be(true), 
        "Same seed should produce identical results"
    end
  end
  
  describe "seeding edge cases (now working)" do
    it "works with seed 0" do
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: 0)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: 0)
      
      labels1 = kmeans1.fit_predict(test_data)
      labels2 = kmeans2.fit_predict(test_data)
      
      expect(labels1).to eq(labels2)
    end
    
    it "works with negative seeds" do
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: -1)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: -1)
      
      labels1 = kmeans1.fit_predict(test_data)
      labels2 = kmeans2.fit_predict(test_data)
      
      expect(labels1).to eq(labels2)
    end
    
    it "produces different results with different seeds" do
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 999)
      
      result1 = kmeans1.fit(test_data)
      result2 = kmeans2.fit(test_data)
      
      # Different seeds should produce different results
      different_centroids = result1.centroids != result2.centroids
      different_labels = result1.labels != result2.labels
      
      expect(different_centroids || different_labels).to be true
    end
    
    it "handles nil random_seed without crashing" do
      kmeans = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: nil)
      expect { kmeans.fit_predict(test_data) }.not_to raise_error
      
      # Without seed, results should be non-deterministic
      result1 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: nil).fit_predict(test_data)
      result2 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: nil).fit_predict(test_data)
      # Note: They might occasionally be the same by chance, so we don't test inequality
    end
    
    it "works with very large seed values" do
      large_seed = 2**32 - 1
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: large_seed)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 2, random_seed: large_seed)
      
      labels1 = kmeans1.fit_predict(test_data)
      labels2 = kmeans2.fit_predict(test_data)
      
      expect(labels1).to eq(labels2)
    end
    
    it "should be reproducible across separate process runs" do
      skip "Would require multi-process testing - documented for future implementation"
    end
  end
  
  describe "comparison with other algorithms" do
    it "now behaves like UMAP seeding - both work!" do
      # UMAP has working seeding - KMeans should behave similarly
      umap_data = test_data.first(15) # UMAP needs fewer points
      
      # UMAP with seeding works
      umap1 = ClusterKit::Dimensionality::UMAP.new(random_seed: 42, n_neighbors: 5)
      umap2 = ClusterKit::Dimensionality::UMAP.new(random_seed: 42, n_neighbors: 5)
      
      umap_result1 = umap1.fit_transform(umap_data)
      umap_result2 = umap2.fit_transform(umap_data)
      
      expect(umap_result1).to eq(umap_result2), "UMAP seeding should work"
      
      # KMeans seeding now also works!
      kmeans1 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
      kmeans2 = ClusterKit::Clustering::KMeans.new(k: 3, random_seed: 42)
      
      kmeans_labels1 = kmeans1.fit_predict(test_data)
      kmeans_labels2 = kmeans2.fit_predict(test_data)
      
      expect(kmeans_labels1).to eq(kmeans_labels2), 
        "KMeans seeding now works like UMAP"
    end
  end
end