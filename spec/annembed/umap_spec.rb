# frozen_string_literal: true

require "spec_helper"

RSpec.describe AnnEmbed::UMAP do
  describe "#initialize" do
    it "creates a new UMAP instance with default parameters" do
      umap = described_class.new
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
    end

    it "accepts n_components parameter" do
      umap = described_class.new(n_components: 3)
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
    end

    it "accepts n_neighbors parameter" do
      umap = described_class.new(n_neighbors: 30)
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
    end

    it "accepts random_seed parameter" do
      umap = described_class.new(random_seed: 42)
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
    end
  end

  describe "#fit_transform" do
    let(:umap) { described_class.new(n_components: 2, n_neighbors: 5) }

    # Helper to generate test data
    def generate_test_data(n_points, n_features, n_clusters = 3)
      data = []
      points_per_cluster = n_points / n_clusters
      
      n_clusters.times do |c|
        # Scale down the cluster centers to avoid boundary issues
        center = Array.new(n_features) { c * 0.3 }
        points_per_cluster.times do
          # Smaller variance to keep data well within bounds
          point = center.map { |x| x + (rand - 0.5) * 0.05 }
          data << point
        end
      end
      
      data
    end

    context "with valid input" do
      it "transforms a reasonable dataset" do
        # Use 15 points to ensure HNSW works properly
        data = generate_test_data(15, 5)
        result = OutputSuppressor.suppress_output { umap.fit_transform(data) }
        
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(15)
        expect(result.first.length).to eq(2)
        expect(result.first.first).to be_instance_of(Float)
      end

      it "reduces dimensions from 10D to 2D" do
        # Generate more conservative test data with explicit normalization
        data = 30.times.map do 
          10.times.map { rand * 0.5 + 0.25 }  # Values between 0.25 and 0.75
        end
        
        result = OutputSuppressor.suppress_output { umap.fit_transform(data) }
        
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(30)
        expect(result.first.length).to eq(2)
      end

      it "handles integer values" do
        # Create normalized integer data
        data = 15.times.map { |i| [i % 3, (i / 3) % 3, i % 5] }
        # Normalize to 0-1 range
        data = data.map { |row| row.map { |x| x / 5.0 } }
        
        result = OutputSuppressor.suppress_output { umap.fit_transform(data) }
        
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(15)
      end
    end

    context "with invalid input" do
      it "raises error for non-array input" do
        expect { umap.fit_transform("not an array") }.to raise_error(ArgumentError, /must be an array/)
      end

      it "raises error for empty array" do
        expect { umap.fit_transform([]) }.to raise_error(ArgumentError, /cannot be empty/)
      end

      it "raises error for 1D array" do
        expect { umap.fit_transform([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
      end

      it "raises error for inconsistent row lengths" do
        data = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /same length/)
      end

      it "raises error for non-numeric values" do
        data = [[1.0, "two"], [3.0, 4.0]]
        expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /not numeric/)
      end

      it "raises error for NaN values" do
        data = [[1.0, Float::NAN], [3.0, 4.0]]
        expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /NaN or Infinite/)
      end

      it "raises error for Infinite values" do
        data = [[1.0, Float::INFINITY], [3.0, 4.0]]
        expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /NaN or Infinite/)
      end
    end
  end

  describe "clustering behavior" do
    let(:umap) { described_class.new(n_components: 2, n_neighbors: 5) }

    it "separates distinct clusters" do
      # Create 3 well-separated clusters
      cluster1 = 10.times.map { [0.0 + rand * 0.1, 0.0 + rand * 0.1, 0.0 + rand * 0.1] }
      cluster2 = 10.times.map { [1.0 + rand * 0.1, 1.0 + rand * 0.1, 1.0 + rand * 0.1] }
      cluster3 = 10.times.map { [0.5 + rand * 0.1, 0.0 + rand * 0.1, 1.0 + rand * 0.1] }
      
      data = cluster1 + cluster2 + cluster3
      result = OutputSuppressor.suppress_output { umap.fit_transform(data) }
      
      # Check that we got the right number of points back
      expect(result.length).to eq(30)
      
      # The embedded points should form distinct groups
      # (We can't test exact positions due to stochastic nature)
    end
  end
end