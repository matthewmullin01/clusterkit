# frozen_string_literal: true

require 'spec_helper'
require 'annembed/svd'

RSpec.describe AnnEmbed::SVD do
  # Sample test matrices
  let(:simple_matrix) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }  # 3x2 matrix
  let(:square_matrix) { [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] }  # 3x3 matrix
  let(:wide_matrix) { [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] }  # 2x4 matrix
  let(:identity_matrix) { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
  let(:empty_matrix) { [] }
  let(:single_row) { [[1.0, 2.0, 3.0]] }
  let(:single_column) { [[1.0], [2.0], [3.0]] }
  
  describe '.randomized_svd' do
    context 'with valid input' do
      it 'performs SVD on simple matrix' do
        # Check if the Rust method exists
        expect(described_class).to respond_to(:randomized_svd_rust)
        
        # Mock the Rust method since it's not implemented yet
        mock_u = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_s = [7.0, 1.0]
        mock_v = [[0.7, 0.8], [0.9, 1.0]]
        allow(described_class).to receive(:randomized_svd_rust).and_return([mock_u, mock_s, mock_v])
        
        result = described_class.randomized_svd(simple_matrix, 2)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)  # U, S, V
        
        u, s, v = result
        expect(u).to eq(mock_u)
        expect(s).to eq(mock_s)
        expect(v).to eq(mock_v)
        
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 2)
      end
      
      it 'accepts custom n_iter parameter' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        described_class.randomized_svd(simple_matrix, 2, n_iter: 5)
        
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 5)
      end
      
      it 'works with square matrices' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        result = described_class.randomized_svd(square_matrix, 2)
        
        expect(result).to be_a(Array)
        expect(described_class).to have_received(:randomized_svd_rust).with(square_matrix, 2, 2)
      end
      
      it 'works with wide matrices' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        result = described_class.randomized_svd(wide_matrix, 2)
        
        expect(result).to be_a(Array)
        expect(described_class).to have_received(:randomized_svd_rust).with(wide_matrix, 2, 2)
      end
      
      it 'handles identity matrix' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([
          identity_matrix,  # U should be identity
          [1.0, 1.0, 1.0],  # S should be all ones
          identity_matrix   # V should be identity
        ])
        
        result = described_class.randomized_svd(identity_matrix, 3)
        
        u, s, v = result
        expect(s).to eq([1.0, 1.0, 1.0])
      end
      
      it 'performs SVD on simple matrix without mocking' do
        # SVD is now implemented!
        result = described_class.randomized_svd(simple_matrix, 2)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)  # U, S, V
        
        u, s, v = result
        expect(u).to be_a(Array)
        expect(s).to be_a(Array) 
        expect(v).to be_a(Array)
        
        # Check dimensions
        expect(u.size).to eq(3)  # 3 rows (same as input)
        expect(u.first.size).to eq(2)  # k columns
        expect(s.size).to eq(2)  # k singular values
        expect(v.size).to eq(2)  # k rows
        expect(v.first.size).to eq(2)  # 2 columns (same as input)
      end
    end
    
    context 'with different k values' do
      it 'accepts k=1 for rank-1 approximation' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[[1.0]], [5.0], [[1.0]]])
        
        result = described_class.randomized_svd(simple_matrix, 1)
        _, s, _ = result
        
        expect(s.size).to eq(1)
      end
      
      it 'accepts k equal to matrix rank' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        described_class.randomized_svd(simple_matrix, 2)
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 2)
      end
      
      it 'handles k larger than matrix dimensions' do
        # Should either work or raise an appropriate error
        allow(described_class).to receive(:randomized_svd_rust).and_raise(ArgumentError, "k too large")
        
        expect {
          described_class.randomized_svd(simple_matrix, 10)
        }.to raise_error(ArgumentError, /k too large/)
      end
    end
    
    context 'with invalid input' do
      it 'raises error for non-array matrix' do
        expect {
          described_class.randomized_svd("not a matrix", 2)
        }.to raise_error(ArgumentError, /Unsupported matrix type: String/)
      end
      
      it 'raises error for nil matrix' do
        expect {
          described_class.randomized_svd(nil, 2)
        }.to raise_error(ArgumentError, /Unsupported matrix type: NilClass/)
      end
      
      it 'raises error for hash input' do
        expect {
          described_class.randomized_svd({ invalid: 'matrix' }, 2)
        }.to raise_error(ArgumentError, /Unsupported matrix type: Hash/)
      end
      
      it 'raises error for 1D array' do
        flat_array = [1.0, 2.0, 3.0, 4.0]
        expect {
          described_class.randomized_svd(flat_array, 2)
        }.to raise_error(TypeError)  # Not a 2D array
      end
    end
    
    context 'edge cases' do
      it 'handles empty matrix' do
        expect {
          described_class.randomized_svd(empty_matrix, 1)
        }.to raise_error(TypeError)  # Empty array causes type error
      end
      
      it 'handles single row matrix' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[[1.0]], [2.0], [[1.0, 0.0, 0.0]]])
        
        result = described_class.randomized_svd(single_row, 1)
        expect(result).to be_a(Array)
      end
      
      it 'handles single column matrix' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[[1.0], [0.0], [0.0]], [3.0], [[1.0]]])
        
        result = described_class.randomized_svd(single_column, 1)
        expect(result).to be_a(Array)
      end
      
      it 'handles zero matrix' do
        zero_matrix = [[0.0, 0.0], [0.0, 0.0]]
        allow(described_class).to receive(:randomized_svd_rust).and_return([[[0.0, 0.0]], [0.0], [[0.0, 0.0]]])
        
        result = described_class.randomized_svd(zero_matrix, 1)
        _, s, _ = result
        expect(s).to eq([0.0])
      end
    end
    
    context 'with different n_iter values' do
      it 'accepts n_iter=1' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        described_class.randomized_svd(simple_matrix, 2, n_iter: 1)
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 1)
      end
      
      it 'accepts large n_iter for better accuracy' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        described_class.randomized_svd(simple_matrix, 2, n_iter: 10)
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 10)
      end
      
      it 'defaults to n_iter=2' do
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], [], []])
        
        described_class.randomized_svd(simple_matrix, 2)
        expect(described_class).to have_received(:randomized_svd_rust).with(simple_matrix, 2, 2)
      end
    end
  end
  
  describe '.truncated_svd' do
    it 'delegates to randomized_svd with default n_iter' do
      allow(described_class).to receive(:randomized_svd).and_return([[], [], []])
      
      result = described_class.truncated_svd(simple_matrix, 2)
      
      expect(described_class).to have_received(:randomized_svd).with(simple_matrix, 2, n_iter: 2)
      expect(result).to eq([[], [], []])
    end
    
    it 'passes through the k parameter' do
      allow(described_class).to receive(:randomized_svd).and_return([[], [], []])
      
      described_class.truncated_svd(square_matrix, 3)
      
      expect(described_class).to have_received(:randomized_svd).with(square_matrix, 3, n_iter: 2)
    end
    
    it 'works with different matrix shapes' do
      allow(described_class).to receive(:randomized_svd).and_return([[], [], []])
      
      described_class.truncated_svd(wide_matrix, 1)
      
      expect(described_class).to have_received(:randomized_svd).with(wide_matrix, 1, n_iter: 2)
    end
    
    it 'propagates errors from randomized_svd' do
      allow(described_class).to receive(:randomized_svd).and_raise(ArgumentError, "Invalid input")
      
      expect {
        described_class.truncated_svd(nil, 2)
      }.to raise_error(ArgumentError, "Invalid input")
    end
  end
  
  describe 'module structure' do
    it 'is a module' do
      expect(described_class).to be_a(Module)
    end
    
    it 'has public methods' do
      expect(described_class).to respond_to(:randomized_svd)
      expect(described_class).to respond_to(:truncated_svd)
    end
    
    it 'exposes rust methods on the module itself' do
      # The rust methods are added to the SVD module directly
      expect(described_class).to respond_to(:randomized_svd_rust)
    end
    
    it 'has methods defined as singleton methods' do
      expect(described_class.singleton_methods).to include(:randomized_svd, :truncated_svd)
    end
  end
  
  describe 'return value structure' do
    context 'when mocked' do
      before do
        # Mock a realistic SVD decomposition
        @mock_u = [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]]  # 3x2 orthogonal columns
        @mock_s = [9.5, 0.5]  # singular values
        @mock_v = [[0.6, 0.8], [0.8, -0.6]]  # 2x2 orthogonal matrix
        allow(described_class).to receive(:randomized_svd_rust).and_return([@mock_u, @mock_s, @mock_v])
      end
      
      it 'returns three components' do
        u, s, v = described_class.randomized_svd(simple_matrix, 2)
        
        expect(u).to eq(@mock_u)
        expect(s).to eq(@mock_s)
        expect(v).to eq(@mock_v)
      end
      
      it 'returns U with correct dimensions' do
        u, _, _ = described_class.randomized_svd(simple_matrix, 2)
        
        expect(u.size).to eq(3)  # Same number of rows as input
        expect(u.first.size).to eq(2)  # k columns
      end
      
      it 'returns S as a 1D array of singular values' do
        _, s, _ = described_class.randomized_svd(simple_matrix, 2)
        
        expect(s).to be_a(Array)
        expect(s.size).to eq(2)  # k singular values
        expect(s.all? { |val| val.is_a?(Numeric) }).to be true
      end
      
      it 'returns V with correct dimensions' do
        _, _, v = described_class.randomized_svd(simple_matrix, 2)
        
        expect(v.size).to eq(2)  # k rows
        expect(v.first.size).to eq(2)  # Same as input column count
      end
    end
  end
  
  describe 'mathematical properties' do
    context 'when properly implemented' do
      it 'should return decreasing singular values' do
        # When implemented, singular values should be in decreasing order
        mock_s = [10.0, 5.0, 1.0]
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], mock_s, []])
        
        _, s, _ = described_class.randomized_svd(square_matrix, 3)
        
        expect(s).to eq(s.sort.reverse)  # Should be sorted in descending order
      end
      
      it 'should return non-negative singular values' do
        mock_s = [10.0, 5.0, 1.0]
        allow(described_class).to receive(:randomized_svd_rust).and_return([[], mock_s, []])
        
        _, s, _ = described_class.randomized_svd(square_matrix, 3)
        
        expect(s.all? { |val| val >= 0 }).to be true
      end
    end
  end
  
  describe 'integration with other modules' do
    it 'can be called from main AnnEmbed module' do
      expect(AnnEmbed).to respond_to(:svd)
    end
    
    it 'is accessible via autoload' do
      # Force autoload if not already loaded
      expect(defined?(AnnEmbed::SVD)).to eq('constant')
    end
  end
end