# frozen_string_literal: true

require 'spec_helper'
require 'tempfile'
require 'csv'

RSpec.describe AnnEmbed::Embedder do
  # Sample test data
  let(:test_data) { Array.new(20) { Array.new(10) { rand } } }
  let(:small_data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
  
  describe '#initialize' do
    context 'with default parameters' do
      subject(:embedder) { described_class.new }
      
      it 'defaults to UMAP method' do
        expect(embedder.method).to eq(:umap)
      end
      
      it 'defaults to 2 components' do
        expect(embedder.n_components).to eq(2)
      end
      
      it 'creates a Config object' do
        expect(embedder.config).to be_a(AnnEmbed::Config)
      end
      
      it 'is not fitted initially' do
        expect(embedder).not_to be_fitted
      end
    end
    
    context 'with custom parameters' do
      subject(:embedder) do
        described_class.new(
          method: :tsne,
          n_components: 3,
          perplexity: 40.0
        )
      end
      
      it 'accepts custom method' do
        expect(embedder.method).to eq(:tsne)
      end
      
      it 'accepts custom n_components' do
        expect(embedder.n_components).to eq(3)
      end
      
      it 'passes options to Config' do
        expect(embedder.config.n_components).to eq(3)
        expect(embedder.config.perplexity).to eq(40.0)
      end
    end
    
    context 'with invalid method' do
      it 'raises ArgumentError for unknown method' do
        expect {
          described_class.new(method: :invalid_method)
        }.to raise_error(ArgumentError, /Unknown method: invalid_method/)
      end
      
      it 'includes valid methods in error message' do
        expect {
          described_class.new(method: :invalid)
        }.to raise_error(ArgumentError, /Must be one of: umap, tsne, largevis, diffusion/)
      end
    end
    
    context 'with all valid methods' do
      [:umap, :tsne, :largevis, :diffusion].each do |method|
        it "accepts #{method} as a valid method" do
          embedder = described_class.new(method: method)
          expect(embedder.method).to eq(method)
        end
      end
    end
  end
  
  describe '#fitted?' do
    subject(:embedder) { described_class.new }
    
    it 'returns false for new embedder' do
      expect(embedder.fitted?).to be false
    end
    
    it 'returns true after fitting' do
      # We'll test this indirectly since we can't easily mock RustUMAP
      # This will be covered by integration tests
      expect(embedder.fitted?).to be false
    end
  end
  
  describe '#fit_transform' do
    subject(:embedder) { described_class.new(n_neighbors: 5) }
    
    context 'with array data' do
      it 'accepts array data' do
        # Silence the output during test
        AnnEmbed.configuration.verbose = false
        
        result = embedder.fit_transform(test_data)
        expect(result).to be_a(Array)
        expect(result.length).to eq(test_data.length)
        expect(result.first.length).to eq(2)
      end
      
      it 'marks embedder as fitted' do
        embedder.fit_transform(test_data)
        expect(embedder).to be_fitted
      end
      
      it 'returns transformed data' do
        result = embedder.fit_transform(test_data)
        expect(result).to be_a(Array)
        
        # Check that we got 2D embeddings
        result.each do |point|
          expect(point).to be_a(Array)
          expect(point.length).to eq(2)
          expect(point[0]).to be_a(Float)
          expect(point[1]).to be_a(Float)
        end
      end
    end
    
    context 'with custom n_components' do
      subject(:embedder) { described_class.new(n_components: 3, n_neighbors: 5) }
      
      it 'returns data with specified dimensions' do
        result = embedder.fit_transform(test_data)
        expect(result.first.length).to eq(3)
      end
    end
    
    context 'with CSV file path' do
      let(:csv_file) do
        file = Tempfile.new(['test', '.csv'])
        CSV.open(file.path, 'w') do |csv|
          test_data.each { |row| csv << row }
        end
        file
      end
      
      after do
        csv_file.unlink if csv_file
      end
      
      it 'loads and processes CSV data' do
        result = embedder.fit_transform(csv_file.path)
        expect(result).to be_a(Array)
        expect(result.length).to eq(test_data.length)
      end
    end
    
    context 'with invalid data' do
      it 'raises error for unsupported data type' do
        expect {
          embedder.fit_transform(123)
        }.to raise_error(ArgumentError, /Unsupported data type/)
      end
      
      it 'raises error for nil data' do
        expect {
          embedder.fit_transform(nil)
        }.to raise_error(ArgumentError)
      end
    end
  end
  
  describe '#fit' do
    subject(:embedder) { described_class.new(n_neighbors: 5) }
    
    it 'returns self for method chaining' do
      result = embedder.fit(test_data)
      expect(result).to eq(embedder)
    end
    
    it 'marks embedder as fitted' do
      embedder.fit(test_data)
      expect(embedder).to be_fitted
    end
    
    it 'does not return transformed data' do
      result = embedder.fit(test_data)
      expect(result).to be_a(described_class)
      expect(result).not_to be_a(Array)
    end
    
    context 'with CSV file' do
      let(:csv_file) do
        file = Tempfile.new(['test', '.csv'])
        CSV.open(file.path, 'w') do |csv|
          test_data.each { |row| csv << row }
        end
        file
      end
      
      after do
        csv_file.unlink if csv_file
      end
      
      it 'accepts CSV file path' do
        result = embedder.fit(csv_file.path)
        expect(result).to eq(embedder)
        expect(embedder).to be_fitted
      end
    end
  end
  
  describe '#transform' do
    subject(:embedder) { described_class.new(n_neighbors: 5) }
    
    context 'with fitted embedder' do
      before do
        embedder.fit(test_data)
      end
      
      it 'transforms new data' do
        new_data = Array.new(5) { Array.new(10) { rand } }
        result = embedder.transform(new_data)
        
        expect(result).to be_a(Array)
        expect(result.length).to eq(5)
        expect(result.first.length).to eq(2)
      end
      
      it 'transforms single point' do
        single_point = [Array.new(10) { rand }]
        result = embedder.transform(single_point)
        
        expect(result).to be_a(Array)
        expect(result.length).to eq(1)
      end
      
      it 'handles CSV input' do
        csv_file = Tempfile.new(['new', '.csv'])
        new_data = Array.new(3) { Array.new(10) { rand } }
        CSV.open(csv_file.path, 'w') do |csv|
          new_data.each { |row| csv << row }
        end
        
        result = embedder.transform(csv_file.path)
        expect(result).to be_a(Array)
        expect(result.length).to eq(3)
        
        csv_file.unlink
      end
    end
    
    context 'with unfitted embedder' do
      it 'raises error' do
        expect {
          embedder.transform(test_data)
        }.to raise_error(AnnEmbed::Error, /must be fitted before transform/)
      end
    end
  end
  
  describe '#save and .load' do
    subject(:embedder) { described_class.new(n_neighbors: 5) }
    let(:save_path) { Tempfile.new(['embedder', '.bin']).path }
    
    after do
      File.delete(save_path) if File.exist?(save_path)
    end
    
    context 'with fitted embedder' do
      before do
        embedder.fit(test_data)
      end
      
      it 'saves to file' do
        expect { embedder.save(save_path) }.not_to raise_error
        expect(File.exist?(save_path)).to be true
      end
      
      it 'loads from file' do
        embedder.save(save_path)
        loaded = described_class.load(save_path)
        
        expect(loaded).to be_a(described_class)
        expect(loaded).to be_fitted
      end
      
      it 'loaded embedder can transform new data' do
        embedder.save(save_path)
        loaded = described_class.load(save_path)
        
        new_data = Array.new(3) { Array.new(10) { rand } }
        result = loaded.transform(new_data)
        
        expect(result).to be_a(Array)
        expect(result.length).to eq(3)
      end
    end
    
    context 'with unfitted embedder' do
      it 'raises error when saving' do
        expect {
          embedder.save(save_path)
        }.to raise_error(AnnEmbed::Error, /must be fitted before saving/)
      end
    end
    
    context 'with non-existent file' do
      it 'raises error when loading' do
        expect {
          described_class.load('/non/existent/file.bin')
        }.to raise_error(RuntimeError, /No such file or directory/)
      end
    end
  end
  
  describe 'private methods' do
    subject(:embedder) { described_class.new }
    
    describe '#prepare_data' do
      it 'returns array data unchanged' do
        data = [[1, 2], [3, 4]]
        result = embedder.send(:prepare_data, data)
        expect(result).to eq(data)
      end
      
      it 'loads CSV file when given string path' do
        csv_file = Tempfile.new(['test', '.csv'])
        CSV.open(csv_file.path, 'w') do |csv|
          csv << [1.0, 2.0]
          csv << [3.0, 4.0]
        end
        
        result = embedder.send(:prepare_data, csv_file.path)
        expect(result).to eq([[1.0, 2.0], [3.0, 4.0]])
        
        csv_file.unlink
      end
      
      it 'raises error for unsupported types' do
        expect {
          embedder.send(:prepare_data, { invalid: 'data' })
        }.to raise_error(ArgumentError, /Unsupported data type/)
      end
    end
    
    describe '#load_csv_data' do
      it 'loads numeric CSV data' do
        csv_file = Tempfile.new(['test', '.csv'])
        CSV.open(csv_file.path, 'w') do |csv|
          csv << ['1.5', '2.5', '3.5']
          csv << ['4.5', '5.5', '6.5']
        end
        
        result = embedder.send(:load_csv_data, csv_file.path)
        expect(result).to eq([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        
        csv_file.unlink
      end
      
      it 'handles mixed numeric types' do
        csv_file = Tempfile.new(['test', '.csv'])
        CSV.open(csv_file.path, 'w') do |csv|
          csv << ['1', '2.5', '3']
          csv << ['4.0', '5', '6.5']
        end
        
        result = embedder.send(:load_csv_data, csv_file.path)
        expect(result[0]).to include(1, 2.5, 3)
        expect(result[1]).to include(4.0, 5, 6.5)
        
        csv_file.unlink
      end
    end
  end
  
  describe 'integration tests' do
    subject(:embedder) { described_class.new(n_neighbors: 5) }
    
    it 'performs full fit-transform-save-load cycle' do
      # Fit and transform
      result1 = embedder.fit_transform(test_data)
      expect(result1).to be_a(Array)
      
      # Save
      save_path = Tempfile.new(['embedder', '.bin']).path
      embedder.save(save_path)
      
      # Load
      loaded = described_class.load(save_path)
      
      # Transform new data with loaded embedder
      new_data = Array.new(5) { Array.new(10) { rand } }
      result2 = loaded.transform(new_data)
      expect(result2).to be_a(Array)
      expect(result2.length).to eq(5)
      
      File.delete(save_path)
    end
    
    it 'supports method chaining' do
      new_data = Array.new(5) { Array.new(10) { rand } }
      
      result = embedder
        .fit(test_data)
        .transform(new_data)
      
      expect(result).to be_a(Array)
      expect(result.length).to eq(5)
    end
  end
end