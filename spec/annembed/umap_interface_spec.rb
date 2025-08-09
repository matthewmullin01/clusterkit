# frozen_string_literal: true

require "spec_helper"
require "tempfile"
require "json"

RSpec.describe "AnnEmbed::UMAP interface" do
  let(:test_data) { 15.times.map { 5.times.map { rand * 0.1 - 0.05 } } }
  let(:new_data) { 3.times.map { 5.times.map { rand * 0.1 - 0.05 } } }
  
  describe "initialization" do
    it "creates a new UMAP instance with default parameters" do
      umap = AnnEmbed::UMAP.new
      expect(umap).to be_instance_of(AnnEmbed::UMAP)
      expect(umap.n_components).to eq(2)
      expect(umap.n_neighbors).to eq(15)
      expect(umap.random_seed).to be_nil
    end
    
    it "accepts custom parameters" do
      umap = AnnEmbed::UMAP.new(
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
    let(:umap) { AnnEmbed::UMAP.new }
    
    it "returns false for unfitted model" do
      expect(umap.fitted?).to be false
    end
    
    it "returns true after fit" do
      OutputSuppressor.suppress_output { umap.fit(test_data) }
      expect(umap.fitted?).to be true
    end
    
    it "returns true after fit_transform" do
      umap2 = AnnEmbed::UMAP.new
      OutputSuppressor.suppress_output { umap2.fit_transform(test_data) }
      expect(umap2.fitted?).to be true
    end
  end
  
  describe "#fit" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }
    
    it "returns self for method chaining" do
      result = OutputSuppressor.suppress_output { umap.fit(test_data) }
      expect(result).to be(umap)
    end
    
    it "marks the model as fitted" do
      OutputSuppressor.suppress_output { umap.fit(test_data) }
      expect(umap.fitted?).to be true
    end
    
    it "raises error for invalid input" do
      expect { umap.fit("not an array") }.to raise_error(ArgumentError, /must be an array/)
      expect { umap.fit([]) }.to raise_error(ArgumentError, /cannot be empty/)
      expect { umap.fit([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
    end
  end
  
  describe "#transform" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }
    
    context "with unfitted model" do
      it "raises error" do
        expect { umap.transform(new_data) }.to raise_error(RuntimeError, /Model must be fitted/)
      end
    end
    
    context "with fitted model" do
      before do
        OutputSuppressor.suppress_output { umap.fit(test_data) }
      end
      
      it "transforms new data" do
        result = OutputSuppressor.suppress_output { umap.transform(new_data) }
        expect(result).to be_instance_of(Array)
        expect(result.length).to eq(3)
        expect(result.first.length).to eq(2)
        expect(result.first.first).to be_instance_of(Float)
      end
      
      it "transforms multiple batches consistently" do
        result1 = OutputSuppressor.suppress_output { umap.transform(new_data) }
        result2 = OutputSuppressor.suppress_output { umap.transform(new_data) }
        
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
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }
    
    it "fits and transforms in one step" do
      result = OutputSuppressor.suppress_output { umap.fit_transform(test_data) }
      
      expect(result).to be_instance_of(Array)
      expect(result.length).to eq(15)
      expect(result.first.length).to eq(2)
      expect(umap.fitted?).to be true
    end
    
    it "produces same results as fit then transform" do
      umap1 = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)
      umap2 = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)
      
      # Method 1: fit_transform
      result1 = OutputSuppressor.suppress_output { umap1.fit_transform(test_data) }
      
      # Method 2: fit then transform
      OutputSuppressor.suppress_output { umap2.fit(test_data) }
      result2 = OutputSuppressor.suppress_output { umap2.transform(test_data) }
      
      # Results should be the same
      result1.each_with_index do |point1, i|
        point2 = result2[i]
        point1.each_with_index do |val1, j|
          expect(val1).to be_within(0.0001).of(point2[j])
        end
      end
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
        expect { umap.save(model_path) }.to raise_error(RuntimeError, /No model to save/)
      end
      
      it "saves fitted model" do
        OutputSuppressor.suppress_output { umap.fit(test_data) }
        umap.save(model_path)
        
        expect(File.exist?(model_path)).to be true
        expect(File.size(model_path)).to be > 0
      end
      
      it "creates directory if needed" do
        nested_path = File.join(Dir.tmpdir, "test_umap_#{Time.now.to_i}", "model.bin")
        OutputSuppressor.suppress_output { umap.fit(test_data) }
        umap.save(nested_path)
        
        expect(File.exist?(nested_path)).to be true
        
        # Clean up
        FileUtils.rm_rf(File.dirname(nested_path))
      end
    end
    
    describe ".load" do
      it "raises error for non-existent file" do
        expect { AnnEmbed::UMAP.load("nonexistent.bin") }.to raise_error(ArgumentError, /File not found/)
      end
      
      it "loads saved model" do
        # Train and save
        OutputSuppressor.suppress_output { umap.fit_transform(test_data) }
        umap.save(model_path)
        
        # Load
        loaded_umap = AnnEmbed::UMAP.load(model_path)
        
        expect(loaded_umap).to be_instance_of(AnnEmbed::UMAP)
        expect(loaded_umap.fitted?).to be true
      end
      
      it "loaded model can transform new data" do
        # Train and save
        original_result = OutputSuppressor.suppress_output { umap.fit_transform(test_data) }
        umap.save(model_path)
        
        # Load and transform
        loaded_umap = AnnEmbed::UMAP.load(model_path)
        new_result = OutputSuppressor.suppress_output { loaded_umap.transform(new_data) }
        
        expect(new_result).to be_instance_of(Array)
        expect(new_result.length).to eq(3)
        expect(new_result.first.length).to eq(2)
      end
      
      it "loaded model produces consistent results" do
        # Train and save
        OutputSuppressor.suppress_output { umap.fit(test_data) }
        original_transform = OutputSuppressor.suppress_output { umap.transform(new_data) }
        umap.save(model_path)
        
        # Load and transform
        loaded_umap = AnnEmbed::UMAP.load(model_path)
        loaded_transform = OutputSuppressor.suppress_output { loaded_umap.transform(new_data) }
        
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
        AnnEmbed::UMAP.export_data(data, data_path)
        
        expect(File.exist?(data_path)).to be true
        content = JSON.parse(File.read(data_path))
        expect(content).to eq(data)
      end
      
      it "creates readable JSON" do
        AnnEmbed::UMAP.export_data(data, data_path)
        content = File.read(data_path)
        
        expect(content).to include("\n") # Pretty printed
        expect(content).to include("  ") # Indented
      end
    end
    
    describe ".import_data" do
      it "imports data from JSON" do
        File.write(data_path, JSON.generate(data))
        imported = AnnEmbed::UMAP.import_data(data_path)
        
        expect(imported).to eq(data)
      end
      
      it "handles pretty-printed JSON" do
        File.write(data_path, JSON.pretty_generate(data))
        imported = AnnEmbed::UMAP.import_data(data_path)
        
        expect(imported).to eq(data)
      end
    end
    
    describe "roundtrip" do
      it "preserves data through export/import" do
        umap = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5)
        result = OutputSuppressor.suppress_output { umap.fit_transform(test_data) }
        
        # Export
        AnnEmbed::UMAP.export_data(result, data_path)
        
        # Import
        imported = AnnEmbed::UMAP.import_data(data_path)
        
        # Should be identical
        expect(imported).to eq(result)
      end
    end
  end
  
  describe "edge cases and error handling" do
    let(:umap) { AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 5) }
    
    it "handles minimum viable dataset" do
      # UMAP needs at least n_neighbors + 1 points
      min_data = 6.times.map { 3.times.map { rand * 0.1 - 0.05 } }
      
      result = OutputSuppressor.suppress_output { umap.fit_transform(min_data) }
      expect(result.length).to eq(6)
    end
    
    it "handles high-dimensional data" do
      high_dim_data = 20.times.map { 100.times.map { rand * 0.1 - 0.05 } }
      
      result = OutputSuppressor.suppress_output { umap.fit_transform(high_dim_data) }
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
  end
end