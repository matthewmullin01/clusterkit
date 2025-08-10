# frozen_string_literal: true

require "spec_helper"
require "tempfile"
require "json"

RSpec.describe "AnnEmbed::UMAP interface" do
  # Use data with structure to avoid degenerate initialization
  # Pure random data can cause all points to initialize to the same location
  let(:test_data) do
    # Create data with some structure - add a small trend to avoid uniform randomness
    15.times.map do |i| 
      30.times.map do |j|
        base = (i.to_f / 15) * 0.01  # Small trend based on row
        noise = rand * 0.01 - 0.005   # Small noise
        base + noise
      end
    end
  end
  let(:new_data) { 3.times.map { |i| 30.times.map { |j| (i.to_f / 3) * 0.01 + rand * 0.01 - 0.005 } } }

  describe "initialization" do
    it "creates a new UMAP instance with default parameters" do
      spec_start_time = Time.now
      puts "Starting spec: creates a new UMAP instance with default parameters"
      umap = AnnEmbed::UMAP.new
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
      expect(umap.n_components).to eq(2)
      expect(umap.n_neighbors).to eq(15)
      expect(umap.random_seed).to be_nil
      puts "Finishing spec: creates a new UMAP instance with default parameters - Time taken: #{Time.now - spec_start_time}"
    end

    it "accepts custom parameters" do
      spec_start_time = Time.now
      puts "Starting spec: accepts custom parameters"
      umap = AnnEmbed::UMAP.new(
        n_components: 3,
        n_neighbors: 10,
        random_seed: 42
      )
      expect(umap.n_components).to eq(3)
      expect(umap.n_neighbors).to eq(10)
      expect(umap.random_seed).to eq(42)
      puts "Finishing spec: accepts custom parameters - Time taken: #{Time.now - spec_start_time}"
    end
  end

  describe "#fitted?" do
    # Don't override the default parameters - the algorithm is sensitive to them
    let(:umap) { AnnEmbed::UMAP.new }

    it "returns false for unfitted model" do
      spec_start_time = Time.now
      puts "Starting spec: returns false for unfitted model"
      expect(umap.fitted?).to be false
      puts "Finishing spec: returns false for unfitted model - Time taken: #{Time.now - spec_start_time}"
    end

    it "returns true after fit" do
      spec_start_time = Time.now
      puts "Starting spec: returns true after fit"
      umap.fit(test_data)
      puts "Middle spec: returns true after fit - Time taken: #{Time.now - spec_start_time}"
      expect(umap.fitted?).to be true
      puts "Finishing spec: returns true after fit - Time taken: #{Time.now - spec_start_time}"
    end

    it "returns true after fit_transform" do
      spec_start_time = Time.now
      puts "Starting spec: returns true after fit_transform"
      umap2 = AnnEmbed::UMAP.new
      umap2.fit_transform(test_data)
      expect(umap2.fitted?).to be true
      puts "Finishing spec: returns true after fit_transform - Time taken: #{Time.now - spec_start_time}"
    end
  end

  describe "#fit" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "returns self for method chaining" do
      spec_start_time = Time.now
      puts "Starting spec: returns self for method chaining"
      result = umap.fit(test_data)
      expect(result).to be(umap)
      puts "Finishing spec: returns self for method chaining - Time taken: #{Time.now - spec_start_time}"
    end

    it "marks the model as fitted" do
      spec_start_time = Time.now
      puts "Starting spec: marks the model as fitted"
      umap.fit(test_data)
      expect(umap.fitted?).to be true
      puts "Finishing spec: marks the model as fitted - Time taken: #{Time.now - spec_start_time}"
    end

    it "raises error for invalid input" do
      spec_start_time = Time.now
      puts "Starting spec: raises error for invalid input"
      expect { umap.fit("not an array") }.to raise_error(ArgumentError, /must be an array/)
      expect { umap.fit([]) }.to raise_error(ArgumentError, /cannot be empty/)
      expect { umap.fit([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
      puts "Finishing spec: raises error for invalid input - Time taken: #{Time.now - spec_start_time}"
    end
  end

  describe "#transform" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }

    context "with unfitted model" do
      it "raises error" do
        spec_start_time = Time.now
        puts "Starting spec: raises error"
        expect { umap.transform(new_data) }.to raise_error(RuntimeError, /Model must be fitted/)
        puts "Finishing spec: raises error - Time taken: #{Time.now - spec_start_time}"
      end
    end

    context "with fitted model" do
      before do
        umap.fit(test_data)
      end

      it "transforms new data" do
        spec_start_time = Time.now
        puts "Starting spec: transforms new data"
        result = umap.transform(new_data)
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(3)
        expect(result.first.length).to eq(2)
        expect(result.first.first).to be_instance_of(Float)
        puts "Finishing spec: transforms new data - Time taken: #{Time.now - spec_start_time}"
      end

      it "transforms multiple batches consistently" do
        spec_start_time = Time.now
        puts "Starting spec: transforms multiple batches consistently"
        result1 = umap.transform(new_data)
        result2 = umap.transform(new_data)

        # Same input should give same output
        result1.each_with_index do |point1, i|
          point2 = result2[i]
          point1.each_with_index do |val1, j|
            expect(val1).to be_within(0.0001).of(point2[j])
          end
        end
        puts "Finishing spec: transforms multiple batches consistently - Time taken: #{Time.now - spec_start_time}"
      end
    end
  end

  describe "#fit_transform" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "fits and transforms in one step" do
      spec_start_time = Time.now
      puts "Starting spec: fits and transforms in one step"
      result = umap.fit_transform(test_data)

      expect(result).to be_instance_of(Array)
      expect(result.length).to eq(15)
      expect(result.first.length).to eq(2)
      expect(umap.fitted?).to be true
      puts "Finishing spec: fits and transforms in one step - Time taken: #{Time.now - spec_start_time}"
    end

    it "produces same results as fit then transform" do
      spec_start_time = Time.now
      puts "Starting spec: produces same results as fit then transform"
      umap1 = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)
      umap2 = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)

      # Method 1: fit_transform
      result1 = umap1.fit_transform(test_data)

      # Method 2: fit then transform
      umap2.fit(test_data)
      result2 = umap2.transform(test_data)

      # Results should be the same
      result1.each_with_index do |point1, i|
        point2 = result2[i]
        point1.each_with_index do |val1, j|
          expect(val1).to be_within(0.0001).of(point2[j])
        end
      end
      puts "Finishing spec: produces same results as fit then transform - Time taken: #{Time.now - spec_start_time}"
    end
  end

  describe "model persistence" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }
    let(:model_path) { Tempfile.new(['umap_model', '.bin']).path }

    after do
      File.delete(model_path) if File.exist?(model_path)
    end

    describe "#save" do
      it "raises error for unfitted model" do
        spec_start_time = Time.now
        puts "Starting spec: raises error for unfitted model"
        expect { umap.save(model_path) }.to raise_error(RuntimeError, /No model to save/)
        puts "Finishing spec: raises error for unfitted model - Time taken: #{Time.now - spec_start_time}"
      end

      it "saves fitted model" do
        spec_start_time = Time.now
        puts "Starting spec: saves fitted model"
        umap.fit(test_data)
        umap.save(model_path)

        expect(File.exist?(model_path)).to be true
        expect(File.size(model_path)).to be > 0
        puts "Finishing spec: saves fitted model - Time taken: #{Time.now - spec_start_time}"
      end

      it "creates directory if needed" do
        spec_start_time = Time.now
        puts "Starting spec: creates directory if needed"
        nested_path = File.join(Dir.tmpdir, "test_umap_#{Time.now.to_i}", "model.bin")
        umap.fit(test_data)
        umap.save(nested_path)

        expect(File.exist?(nested_path)).to be true

        # Clean up
        FileUtils.rm_rf(File.dirname(nested_path))
        puts "Finishing spec: creates directory if needed - Time taken: #{Time.now - spec_start_time}"
      end
    end

    describe ".load" do
      it "raises error for non-existent file" do
        spec_start_time = Time.now
        puts "Starting spec: raises error for non-existent file"
        expect { AnnEmbed::UMAP.load("nonexistent.bin") }.to raise_error(ArgumentError, /File not found/)
        puts "Finishing spec: raises error for non-existent file - Time taken: #{Time.now - spec_start_time}"
      end

      it "loads saved model" do
        spec_start_time = Time.now
        puts "Starting spec: loads saved model"
        # Train and save
        umap.fit_transform(test_data)
        umap.save(model_path)

        # Load
        loaded_umap = AnnEmbed::UMAP.load(model_path)

        expect(loaded_umap).to be_instance_of(AnnEmbed::UMAP)
        expect(loaded_umap.fitted?).to be true
        puts "Finishing spec: loads saved model - Time taken: #{Time.now - spec_start_time}"
      end

      it "loaded model can transform new data" do
        spec_start_time = Time.now
        puts "Starting spec: loaded model can transform new data"
        # Train and save
        original_result = umap.fit_transform(test_data)
        umap.save(model_path)

        # Load and transform
        loaded_umap = AnnEmbed::UMAP.load(model_path)
        new_result = loaded_umap.transform(new_data)

        expect(new_result).to be_instance_of(Array)
        expect(new_result.length).to eq(3)
        expect(new_result.first.length).to eq(2)
        puts "Finishing spec: loaded model can transform new data - Time taken: #{Time.now - spec_start_time}"
      end

      it "loaded model produces consistent results" do
        spec_start_time = Time.now
        puts "Starting spec: loaded model produces consistent results"
        # Train and save
        umap.fit(test_data)
        original_transform = umap.transform(new_data)
        umap.save(model_path)

        # Load and transform
        loaded_umap = AnnEmbed::UMAP.load(model_path)
        loaded_transform = loaded_umap.transform(new_data)

        # Results should be the same
        original_transform.each_with_index do |point1, i|
          point2 = loaded_transform[i]
          point1.each_with_index do |val1, j|
            expect(val1).to be_within(0.0001).of(point2[j])
          end
        end
        puts "Finishing spec: loaded model produces consistent results - Time taken: #{Time.now - spec_start_time}"
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
        spec_start_time = Time.now
        puts "Starting spec: exports data to JSON"
        AnnEmbed::UMAP.export_data(data, data_path)

        expect(File.exist?(data_path)).to be true
        content = JSON.parse(File.read(data_path))
        expect(content).to eq(data)
        puts "Finishing spec: exports data to JSON - Time taken: #{Time.now - spec_start_time}"
      end

      it "creates readable JSON" do
        spec_start_time = Time.now
        puts "Starting spec: creates readable JSON"
        AnnEmbed::UMAP.export_data(data, data_path)
        content = File.read(data_path)

        expect(content).to include("\n") # Pretty printed
        expect(content).to include("  ") # Indented
        puts "Finishing spec: creates readable JSON - Time taken: #{Time.now - spec_start_time}"
      end
    end

    describe ".import_data" do
      it "imports data from JSON" do
        spec_start_time = Time.now
        puts "Starting spec: imports data from JSON"
        File.write(data_path, JSON.generate(data))
        imported = AnnEmbed::UMAP.import_data(data_path)

        expect(imported).to eq(data)
        puts "Finishing spec: imports data from JSON - Time taken: #{Time.now - spec_start_time}"
      end

      it "handles pretty-printed JSON" do
        spec_start_time = Time.now
        puts "Starting spec: handles pretty-printed JSON"
        File.write(data_path, JSON.pretty_generate(data))
        imported = AnnEmbed::UMAP.import_data(data_path)

        expect(imported).to eq(data)
        puts "Finishing spec: handles pretty-printed JSON - Time taken: #{Time.now - spec_start_time}"
      end
    end

    describe "roundtrip" do
      it "preserves data through export/import" do
        spec_start_time = Time.now
        puts "Starting spec: preserves data through export/import"
        umap = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)
        result = umap.fit_transform(test_data)

        # Export
        AnnEmbed::UMAP.export_data(result, data_path)

        # Import
        imported = AnnEmbed::UMAP.import_data(data_path)

        # Should be identical
        expect(imported).to eq(result)
        puts "Finishing spec: preserves data through export/import - Time taken: #{Time.now - spec_start_time}"
      end
    end
  end

  describe "edge cases and error handling" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }

    it "handles minimum viable dataset" do
      spec_start_time = Time.now
      puts "Starting spec: handles minimum viable dataset"
      # UMAP needs at least n_neighbors + 1 points
      min_data = 6.times.map { 10.times.map { rand * 0.02 - 0.01 } }

      result = umap.fit_transform(min_data)
      expect(result.length).to eq(6)
      puts "Finishing spec: handles minimum viable dataset - Time taken: #{Time.now - spec_start_time}"
    end

    it "handles high-dimensional data" do
      spec_start_time = Time.now
      puts "Starting spec: handles high-dimensional data"
      high_dim_data = 20.times.map { 100.times.map { rand * 0.02 - 0.01 } }

      result = umap.fit_transform(high_dim_data)
      expect(result.first.length).to eq(2)
      puts "Finishing spec: handles high-dimensional data - Time taken: #{Time.now - spec_start_time}"
    end

    it "validates data consistency" do
      spec_start_time = Time.now
      puts "Starting spec: validates data consistency"
      inconsistent_data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0]  # Different length
      ]

      expect { umap.fit(inconsistent_data) }.to raise_error(ArgumentError, /same length/)
      puts "Finishing spec: validates data consistency - Time taken: #{Time.now - spec_start_time}"
    end

    it "validates numeric data" do
      spec_start_time = Time.now
      puts "Starting spec: validates numeric data"
      non_numeric_data = [
        [1.0, "two"],
        [3.0, 4.0]
      ]

      expect { umap.fit(non_numeric_data) }.to raise_error(ArgumentError, /not numeric/)
      puts "Finishing spec: validates numeric data - Time taken: #{Time.now - spec_start_time}"
    end
  end
end