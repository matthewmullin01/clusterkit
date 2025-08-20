# frozen_string_literal: true

require "spec_helper"
require "tempfile"
require "json"

RSpec.describe "ClusterKit::Dimensionality::UMAP interface" do
  # Use real embeddings from fixtures if available, otherwise fall back to structured data
  let(:test_data) do
    if fixtures_available?
      # Use real embeddings - these won't cause hanging issues
      load_embedding_fixture('basic_15')
    else
      # Fall back to structured test data (better than pure random)
      warn "Using generated test data. Run 'rake fixtures:generate_embeddings' for better tests."
      generate_structured_test_data(15, 30)
    end
  end

  let(:new_data) do
    if fixtures_available?
      # Use a subset of different embeddings for transform testing
      load_embedding_subset('minimal_6', 3)
    else
      generate_structured_test_data(3, 30)
    end
  end

  describe "initialization" do
    it "creates a new UMAP instance with default parameters" do
      umap = ClusterKit::Dimensionality::UMAP.new
      expect(umap).to be_instance_of(ClusterKit::Dimensionality::UMAP)
      expect(umap.n_components).to eq(2)
      expect(umap.n_neighbors).to eq(15)
      expect(umap.random_seed).to be_nil
    end

    it "accepts custom parameters" do
      umap = ClusterKit::Dimensionality::UMAP.new(
        n_components: 3,
        n_neighbors: 10,
        random_seed: 42
      )
      expect(umap.n_components).to eq(3)
      expect(umap.n_neighbors).to eq(10)
      expect(umap.random_seed).to eq(42)
    end
  end

  describe "#fitted?" do
    # Use n_neighbors appropriate for the data size (15 points)
    # Default of 15 neighbors with 15 points causes degenerate behavior
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_neighbors: 5) }

    it "returns false for unfitted model" do
      expect(umap.fitted?).to be false
    end

    it "returns true after fit" do
      umap.fit(test_data)
      expect(umap.fitted?).to be true
    end

    it "returns true after fit_transform" do
      umap2 = ClusterKit::Dimensionality::UMAP.new(n_neighbors: 5)
      umap2.fit_transform(test_data)
      expect(umap2.fitted?).to be true
    end
  end

  describe "#fit" do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "returns self for method chaining" do
      result = umap.fit(test_data)
      expect(result).to be(umap)
    end

    it "marks the model as fitted" do
      umap.fit(test_data)
      expect(umap.fitted?).to be true
    end

    it "raises error for invalid input" do
      expect { umap.fit("not an array") }.to raise_error(ArgumentError, /must be an array/)
      expect { umap.fit([]) }.to raise_error(ArgumentError, /cannot be empty/)
      expect { umap.fit([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
    end
  end

  describe "#transform" do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5) }

    context "with unfitted model" do
      it "raises error" do
        expect { umap.transform(new_data) }.to raise_error(RuntimeError, /Model must be fitted/)
      end
    end

    context "with fitted model" do
      before do
        umap.fit(test_data)
      end

      it "transforms new data" do
        result = umap.transform(new_data)
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(3)
        expect(result.first.length).to eq(2)
        expect(result.first.first).to be_instance_of(Float)
      end

      it "transforms multiple batches consistently" do
        result1 = umap.transform(new_data)
        result2 = umap.transform(new_data)

        # Same input should give same output
        result1.each_with_index do |point1, i|
          point2 = result2[i]
          point1.each_with_index do |val1, j|
            expect(val1).to be_within(0.0001).of(point2[j])
          end
        end
      end
    end
  end

  describe "#fit_transform" do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "fits and transforms in one step" do
      result = umap.fit_transform(test_data)

      expect(result).to be_instance_of(Array)
      expect(result.length).to eq(15)
      expect(result.first.length).to eq(2)
      expect(umap.fitted?).to be true
    end

    it "both fit_transform and fit->transform produce valid embeddings" do
      # Use fixed random seed for reproducibility
      umap1 = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5, random_seed: 42)
      umap2 = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5, random_seed: 42)

      # Method 1: fit_transform
      result1 = umap1.fit_transform(test_data)

      # Method 2: fit then transform
      umap2.fit(test_data)
      result2 = umap2.transform(test_data)

      # Both methods should produce valid embeddings with the right shape
      expect(result1).to be_instance_of(Array)
      expect(result1.length).to eq(15)
      expect(result1.first.length).to eq(2)

      expect(result2).to be_instance_of(Array)
      expect(result2.length).to eq(15)
      expect(result2.first.length).to eq(2)

      # Both should produce embeddings with reasonable spread (not all the same)
      [result1, result2].each_with_index do |result, idx|
        all_values = result.flatten
        min_val = all_values.min
        max_val = all_values.max
        spread = max_val - min_val

        expect(spread).to be > 0.1, "Method #{idx + 1} produced degenerate embeddings with spread #{spread}"

        # Check that not all points are the same
        first_point = result.first
        not_all_same = result.any? { |point|
          point.zip(first_point).any? { |a, b| (a - b).abs > 0.01 }
        }
        expect(not_all_same).to eq(true)
      end
    end
  end

  describe "model persistence" do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5) }
    let(:model_path) { Tempfile.new(['umap_model', '.bin']).path }

    after do
      File.delete(model_path) if File.exist?(model_path)
    end

    describe "#save_model" do
      it "raises error for unfitted model" do
        expect { umap.save_model(model_path) }.to raise_error(RuntimeError, /No model to save/)
      end

      it "saves fitted model" do
        umap.fit(test_data)
        umap.save_model(model_path)

        expect(File.exist?(model_path)).to be true
        expect(File.size(model_path)).to be > 0
      end

      it "creates directory if needed" do
        nested_path = File.join(Dir.tmpdir, "test_umap_#{Time.now.to_i}", "model.bin")
        umap.fit(test_data)
        umap.save_model(nested_path)

        expect(File.exist?(nested_path)).to be true

        # Clean up
        FileUtils.rm_rf(File.dirname(nested_path))
      end
    end

    describe ".load_model" do
      it "raises error for non-existent file" do
        expect { ClusterKit::Dimensionality::UMAP.load_model("nonexistent.bin") }.to raise_error(ArgumentError, /File not found/)
      end

      it "loads saved model" do
        # Train and save
        umap.fit_transform(test_data)
        umap.save_model(model_path)

        # Load
        loaded_umap = ClusterKit::Dimensionality::UMAP.load_model(model_path)

        expect(loaded_umap).to be_instance_of(ClusterKit::Dimensionality::UMAP)
        expect(loaded_umap.fitted?).to be true
      end

      it "loaded model can transform new data" do
        # Train and save
        original_result = umap.fit_transform(test_data)
        umap.save_model(model_path)

        # Load and transform
        loaded_umap = ClusterKit::Dimensionality::UMAP.load_model(model_path)
        new_result = loaded_umap.transform(new_data)

        expect(new_result).to be_instance_of(Array)
        expect(new_result.length).to eq(3)
        expect(new_result.first.length).to eq(2)
      end

      it "loaded model produces consistent results" do
        # Train and save
        umap.fit(test_data)
        original_transform = umap.transform(new_data)
        umap.save_model(model_path)

        # Load and transform
        loaded_umap = ClusterKit::Dimensionality::UMAP.load_model(model_path)
        loaded_transform = loaded_umap.transform(new_data)

        # Results should be the same
        original_transform.each_with_index do |point1, i|
          point2 = loaded_transform[i]
          point1.each_with_index do |val1, j|
            expect(val1).to be_within(0.0001).of(point2[j])
          end
        end
      end
    end
  end

  describe "data export/import utilities" do
    let(:data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
    let(:data_path) { Tempfile.new(['data', '.json']).path }

    after do
      File.delete(data_path) if File.exist?(data_path)
    end

    describe ".export_data" do
      it "exports data to JSON" do
        ClusterKit::Dimensionality::UMAP.save_data(data, data_path)

        expect(File.exist?(data_path)).to be true
        content = JSON.parse(File.read(data_path))
        expect(content).to eq(data)
      end

      it "creates readable JSON" do
        ClusterKit::Dimensionality::UMAP.save_data(data, data_path)
        content = File.read(data_path)

        expect(content).to include("\n") # Pretty printed
        expect(content).to include("  ") # Indented
      end
    end

    describe ".import_data" do
      it "imports data from JSON" do
        File.write(data_path, JSON.generate(data))
        imported = ClusterKit::Dimensionality::UMAP.load_data(data_path)

        expect(imported).to eq(data)
      end

      it "handles pretty-printed JSON" do
        File.write(data_path, JSON.pretty_generate(data))
        imported = ClusterKit::Dimensionality::UMAP.load_data(data_path)

        expect(imported).to eq(data)
      end
    end

    describe "roundtrip" do
      it "preserves data through export/import" do
        umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5)
        result = umap.fit_transform(test_data)

        # Export
        ClusterKit::Dimensionality::UMAP.save_data(result, data_path)

        # Import
        imported = ClusterKit::Dimensionality::UMAP.load_data(data_path)

        # Should be identical
        expect(imported).to eq(result)
      end
    end
  end

  describe "edge cases and error handling" do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "handles minimum viable dataset" do
      # UMAP needs at least 10 points as enforced by validate_input
      # Try to load a larger fixture, or generate test data
      min_data = if fixtures_available? && File.exist?(File.join(File.dirname(__FILE__), '../fixtures/embeddings/minimal_10.json'))
        load_embedding_fixture('minimal_10')
      elsif fixtures_available? && File.exist?(File.join(File.dirname(__FILE__), '../fixtures/embeddings/tweets_20.json'))
        # Use a subset of tweets_20 if available
        load_embedding_fixture('tweets_20').first(10)
      else
        generate_structured_test_data(10, 10)
      end

      # Use n_neighbors=3 for 10 data points to avoid degenerate behavior
      small_umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 3)
      result = small_umap.fit_transform(min_data)
      expect(result.length).to eq(10)
    end

    it "handles high-dimensional data" do
      high_dim_data = if fixtures_available?
        # Real embeddings are already high-dimensional (384D for MiniLM)
        load_embedding_subset('clusters_30', 20)
      else
        generate_structured_test_data(20, 100)
      end

      result = umap.fit_transform(high_dim_data)
      expect(result.first.length).to eq(2)
    end

    it "validates data consistency" do
      inconsistent_data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0]  # Different length
      ]

      expect { umap.fit(inconsistent_data) }.to raise_error(ArgumentError, /same length/)
    end

    it "validates numeric data" do
      non_numeric_data = [
        [1.0, "two"],
        [3.0, 4.0]
      ]

      expect { umap.fit(non_numeric_data) }.to raise_error(ArgumentError, /not numeric/)
    end

    it "handles clustered embedding data (if fixtures available)" do
      if fixtures_available?
        # Load real embeddings with 3 distinct clusters
        clustered_data = load_embedding_fixture('clusters_30')

        umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5)
        result = umap.fit_transform(clustered_data)

        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(30)
        expect(result.first.length).to eq(2)

        # The embeddings should have reasonable values (not all the same)
        values = result.flatten
        min_val = values.min
        max_val = values.max
        range = max_val - min_val

        expect(range).to be > 0.1  # Should have some spread
      else
        skip "Embedding fixtures not available"
      end
    end
  end
end