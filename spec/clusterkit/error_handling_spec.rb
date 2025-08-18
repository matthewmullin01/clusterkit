# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'ClusterKit error handling' do
  describe 'Error class hierarchy' do
    it 'defines base Error class' do
      expect(ClusterKit::Error).to be < StandardError
    end
    
    it 'defines DataError as subclass of Error' do
      expect(ClusterKit::DataError).to be < ClusterKit::Error
    end
    
    it 'defines IsolatedPointError as subclass of DataError' do
      expect(ClusterKit::IsolatedPointError).to be < ClusterKit::DataError
    end
    
    it 'defines DisconnectedGraphError as subclass of DataError' do
      expect(ClusterKit::DisconnectedGraphError).to be < ClusterKit::DataError
    end
    
    it 'defines ConvergenceError as subclass of Error' do
      expect(ClusterKit::ConvergenceError).to be < ClusterKit::Error
    end
    
    it 'defines InsufficientDataError as subclass of DataError' do
      expect(ClusterKit::InsufficientDataError).to be < ClusterKit::DataError
    end
    
    it 'defines InvalidParameterError as subclass of Error' do
      expect(ClusterKit::InvalidParameterError).to be < ClusterKit::Error
    end
  end
  
  describe 'UMAP error handling' do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 15) }
    let(:data) { Array.new(100) { Array.new(50) { rand } } }
    
    # Mock the RustUMAP to simulate various errors
    class MockRustUMAP
      attr_accessor :error_to_raise
      
      def initialize(params = {})
        @error_to_raise = nil
      end
      
      def fit_transform(data)
        raise @error_to_raise if @error_to_raise
        # Return mock result
        Array.new(data.size) { Array.new(2) { rand } }
      end
      
      def fit(data)
        fit_transform(data)
      end
    end
    
    before do
      # Replace the rust UMAP with our mock
      mock_rust = MockRustUMAP.new
      # Set rust_umap to the mock AND prevent it from being recreated
      umap.instance_variable_set(:@rust_umap, mock_rust)
      # Mock the creation method to return our mock instead of creating a new one
      allow(::ClusterKit::RustUMAP).to receive(:new).and_return(mock_rust)
      @mock_rust = mock_rust
    end
    
    context 'when isolated point error occurs' do
      before do
        @mock_rust.error_to_raise = RuntimeError.new("kgraph_from_hnsw_all: graph will not be connected, isolated point at layer 0")
      end
      
      it 'raises IsolatedPointError' do
        expect { umap.fit_transform(data) }.to raise_error(ClusterKit::IsolatedPointError)
      end
      
      it 'includes helpful message about isolated points' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('isolated points')
          expect(error.message).to include('too far from other points')
        end
      end
      
      it 'provides solutions' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('Reduce n_neighbors')
          expect(error.message).to include('Remove outliers')
          expect(error.message).to include('structure')
        end
      end
      
      it 'includes data context' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('100 samples')
          expect(error.message).to include('50 dimensions')
        end
      end
    end
    
    context 'when convergence error occurs' do
      before do
        @mock_rust.error_to_raise = RuntimeError.new("assertion failed: (*f).abs() <= box_size")
      end
      
      it 'raises ConvergenceError' do
        expect { umap.fit_transform(data) }.to raise_error(ClusterKit::ConvergenceError)
      end
      
      it 'includes message about numerical instability' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('numerical instability')
          expect(error.message).to include('failed to converge')
        end
      end
      
      it 'provides normalization solution' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('Normalize your data')
          expect(error.message).to include('ClusterKit::Preprocessing.normalize')
        end
      end
      
      it 'mentions scale issues' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('scale')
          expect(error.message).to include('extreme values')
        end
      end
    end
    
    context 'when n_neighbors is too large' do
      before do
        @mock_rust.error_to_raise = RuntimeError.new("n_neighbors is larger than dataset")
      end
      
      it 'raises InvalidParameterError' do
        expect { umap.fit_transform(data) }.to raise_error(ClusterKit::InvalidParameterError)
      end
      
      it 'includes current n_neighbors value' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('15')  # The n_neighbors value
        end
      end
      
      it 'suggests appropriate value' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('Suggested value')
        end
      end
      
      it 'mentions auto-adjustment issue' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('auto-adjusted')
          expect(error.message).to include('report it')
        end
      end
    end
    
    context 'when unknown error occurs' do
      before do
        @mock_rust.error_to_raise = RuntimeError.new("some unexpected error")
      end
      
      it 'raises generic ClusterKit::Error' do
        expect { umap.fit_transform(data) }.to raise_error(ClusterKit::Error)
      end
      
      it 'includes original error message' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('some unexpected error')
        end
      end
      
      it 'provides generic solutions' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('reducing n_neighbors')
          expect(error.message).to include('Normalize')
          expect(error.message).to include('NaN or infinite')
        end
      end
      
      it 'suggests PCA as alternative' do
        expect { umap.fit_transform(data) }.to raise_error do |error|
          expect(error.message).to include('consider using PCA')
        end
      end
    end
    
    context 'error handling in fit method' do
      before do
        @mock_rust.error_to_raise = RuntimeError.new("isolated point")
      end
      
      it 'also handles errors in fit method' do
        expect { umap.fit(data) }.to raise_error(ClusterKit::IsolatedPointError)
      end
    end
  end
  
  describe 'Input validation errors' do
    let(:umap) { ClusterKit::Dimensionality::UMAP.new }
    
    it 'raises ArgumentError for non-array input' do
      expect { umap.fit_transform("not an array") }.to raise_error(ArgumentError, /must be an array/)
    end
    
    it 'raises ArgumentError for empty input' do
      expect { umap.fit_transform([]) }.to raise_error(ArgumentError, /cannot be empty/)
    end
    
    it 'raises ArgumentError for 1D array' do
      expect { umap.fit_transform([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
    end
    
    it 'raises ArgumentError for inconsistent row lengths' do
      data = [[1, 2], [3, 4, 5]]
      expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /same length/)
    end
    
    it 'raises ArgumentError for non-numeric values' do
      data = [[1, 2], ["a", "b"]]
      expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /not numeric/)
    end
    
    it 'raises ArgumentError for NaN values' do
      data = [[1.0, 2.0], [Float::NAN, 4.0]]
      expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /NaN or Infinite/)
    end
    
    it 'raises ArgumentError for Infinite values' do
      data = [[1.0, 2.0], [Float::INFINITY, 4.0]]
      expect { umap.fit_transform(data) }.to raise_error(ArgumentError, /NaN or Infinite/)
    end
    
    it 'accepts integer values' do
      data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
      # Should not raise an error for integer values
      expect { umap.send(:validate_input, data) }.not_to raise_error
    end
  end
  
  describe 'Error message formatting' do
    it 'uses bullet points for lists' do
      error = ClusterKit::IsolatedPointError.new(<<~MSG)
        Test error
        • Point one
        • Point two
      MSG
      
      expect(error.message).to include('•')
    end
    
    it 'can be rescued as DataError' do
      begin
        raise ClusterKit::IsolatedPointError, "test"
      rescue ClusterKit::DataError => e
        expect(e).to be_a(ClusterKit::IsolatedPointError)
      end
    end
    
    it 'can be rescued as ClusterKit::Error' do
      begin
        raise ClusterKit::ConvergenceError, "test"
      rescue ClusterKit::Error => e
        expect(e).to be_a(ClusterKit::ConvergenceError)
      end
    end
  end
end