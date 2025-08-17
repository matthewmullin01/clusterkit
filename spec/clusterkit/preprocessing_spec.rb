# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit::Preprocessing do
  describe '.normalize' do
    let(:simple_data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
    let(:empty_data) { [] }
    let(:single_row) { [[1.0, 2.0, 3.0]] }
    let(:zero_variance_data) { [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]] }
    
    context 'with standard normalization' do
      it 'normalizes with default method (standard)' do
        result = described_class.normalize(simple_data)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)
        expect(result.first.size).to eq(2)
        
        # Check that mean is approximately 0 and std is approximately 1
        col1_values = result.map { |row| row[0] }
        col1_mean = col1_values.sum / col1_values.size
        expect(col1_mean).to be_within(0.001).of(0.0)
        
        col1_std = Math.sqrt(col1_values.map { |v| (v - col1_mean) ** 2 }.sum / col1_values.size)
        expect(col1_std).to be_within(0.001).of(1.0)
      end
      
      it 'explicitly uses standard method' do
        result = described_class.normalize(simple_data, method: :standard)
        
        # First column: [1, 3, 5] -> mean = 3, std = sqrt((4+0+4)/3) = 1.633
        # Normalized: [(1-3)/1.633, (3-3)/1.633, (5-3)/1.633] = [-1.225, 0, 1.225]
        expect(result[0][0]).to be_within(0.01).of(-1.225)
        expect(result[1][0]).to be_within(0.01).of(0.0)
        expect(result[2][0]).to be_within(0.01).of(1.225)
      end
      
      it 'handles data with zero variance' do
        result = described_class.normalize(zero_variance_data, method: :standard)
        
        # First column has zero variance, should not be modified (divided by 1.0)
        expect(result[0][0]).to eq(0.0)
        expect(result[1][0]).to eq(0.0)
        expect(result[2][0]).to eq(0.0)
        
        # Second column should be normalized normally
        col2_values = result.map { |row| row[1] }
        col2_mean = col2_values.sum / col2_values.size
        expect(col2_mean).to be_within(0.001).of(0.0)
      end
      
      it 'returns empty array for empty data' do
        result = described_class.normalize(empty_data, method: :standard)
        expect(result).to eq([])
      end
      
      it 'handles single row data' do
        result = described_class.normalize(single_row, method: :standard)
        expect(result).to be_a(Array)
        expect(result.size).to eq(1)
        # Single row will have std of 0, so values become 0
        expect(result[0]).to eq([0.0, 0.0, 0.0])
      end
    end
    
    context 'with minmax normalization' do
      it 'normalizes to [0, 1] range' do
        result = described_class.normalize(simple_data, method: :minmax)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)
        
        # First column: [1, 3, 5] -> min=1, max=5, range=4
        # Normalized: [(1-1)/4, (3-1)/4, (5-1)/4] = [0, 0.5, 1]
        expect(result[0][0]).to eq(0.0)
        expect(result[1][0]).to eq(0.5)
        expect(result[2][0]).to eq(1.0)
        
        # Second column: [2, 4, 6] -> min=2, max=6, range=4
        # Normalized: [(2-2)/4, (4-2)/4, (6-2)/4] = [0, 0.5, 1]
        expect(result[0][1]).to eq(0.0)
        expect(result[1][1]).to eq(0.5)
        expect(result[2][1]).to eq(1.0)
      end
      
      it 'handles data with zero range' do
        result = described_class.normalize(zero_variance_data, method: :minmax)
        
        # First column has zero range, should map to 0
        expect(result[0][0]).to eq(0.0)
        expect(result[1][0]).to eq(0.0)
        expect(result[2][0]).to eq(0.0)
        
        # Second column should be normalized to [0, 1]
        expect(result[0][1]).to eq(0.0)
        expect(result[1][1]).to eq(0.5)
        expect(result[2][1]).to eq(1.0)
      end
      
      it 'returns empty array for empty data' do
        result = described_class.normalize(empty_data, method: :minmax)
        expect(result).to eq([])
      end
      
      it 'handles single row data' do
        result = described_class.normalize(single_row, method: :minmax)
        expect(result).to be_a(Array)
        expect(result.size).to eq(1)
        # Single row will have range of 0, so values become 0
        expect(result[0]).to eq([0.0, 0.0, 0.0])
      end
      
      it 'handles negative values correctly' do
        negative_data = [[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]]
        result = described_class.normalize(negative_data, method: :minmax)
        
        # First column: [-2, 0, 2] -> min=-2, max=2, range=4
        # Normalized: [(-2-(-2))/4, (0-(-2))/4, (2-(-2))/4] = [0, 0.5, 1]
        expect(result[0][0]).to eq(0.0)
        expect(result[1][0]).to eq(0.5)
        expect(result[2][0]).to eq(1.0)
      end
    end
    
    context 'with L2 normalization' do
      it 'normalizes rows to unit length' do
        result = described_class.normalize(simple_data, method: :l2)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)
        
        # Check that each row has unit length
        result.each do |row|
          norm = Math.sqrt(row.sum { |val| val ** 2 })
          expect(norm).to be_within(0.001).of(1.0)
        end
      end
      
      it 'handles specific values correctly' do
        data = [[3.0, 4.0]]  # norm = sqrt(9 + 16) = 5
        result = described_class.normalize(data, method: :l2)
        
        expect(result[0][0]).to eq(0.6)  # 3/5
        expect(result[0][1]).to eq(0.8)  # 4/5
      end
      
      it 'handles zero vectors' do
        zero_data = [[0.0, 0.0], [1.0, 0.0]]
        result = described_class.normalize(zero_data, method: :l2)
        
        # Zero vector should remain zero (divided by 1.0 to avoid NaN)
        expect(result[0]).to eq([0.0, 0.0])
        
        # Non-zero vector should be normalized
        expect(result[1]).to eq([1.0, 0.0])
      end
      
      it 'returns empty array for empty data' do
        result = described_class.normalize(empty_data, method: :l2)
        expect(result).to eq([])
      end
      
      it 'handles negative values correctly' do
        data = [[-3.0, 4.0]]  # norm = sqrt(9 + 16) = 5
        result = described_class.normalize(data, method: :l2)
        
        expect(result[0][0]).to eq(-0.6)  # -3/5
        expect(result[0][1]).to eq(0.8)   # 4/5
      end
    end
    
    context 'with invalid input' do
      it 'raises error for unsupported data type' do
        expect {
          described_class.normalize("not an array")
        }.to raise_error(ArgumentError, /Unsupported data type: String/)
      end
      
      it 'raises error for nil data' do
        expect {
          described_class.normalize(nil)
        }.to raise_error(ArgumentError, /Unsupported data type: NilClass/)
      end
      
      it 'raises error for unknown normalization method' do
        expect {
          described_class.normalize(simple_data, method: :unknown)
        }.to raise_error(ArgumentError, /Unknown normalization method: unknown/)
      end
    end
    
    context 'edge cases' do
      it 'handles very large values' do
        large_data = [[1e10, 2e10], [3e10, 4e10]]
        result = described_class.normalize(large_data, method: :standard)
        
        expect(result).to be_a(Array)
        # Should still normalize to mean 0, std 1
        col1_values = result.map { |row| row[0] }
        col1_mean = col1_values.sum / col1_values.size
        expect(col1_mean).to be_within(0.001).of(0.0)
      end
      
      it 'handles very small values' do
        small_data = [[1e-10, 2e-10], [3e-10, 4e-10]]
        result = described_class.normalize(small_data, method: :minmax)
        
        expect(result).to be_a(Array)
        # Should still normalize to [0, 1]
        expect(result[0][0]).to be_within(0.001).of(0.0)
        expect(result[1][0]).to be_within(0.001).of(1.0)
      end
      
      it 'handles mixed positive and negative values' do
        mixed_data = [[-10.0, 5.0], [0.0, -5.0], [10.0, 0.0]]
        result = described_class.normalize(mixed_data, method: :standard)
        
        expect(result).to be_a(Array)
        expect(result.size).to eq(3)
        
        # Check that normalization works correctly
        col1_values = result.map { |row| row[0] }
        col1_mean = col1_values.sum / col1_values.size
        expect(col1_mean).to be_within(0.001).of(0.0)
      end
    end
  end
  
  describe '.pca_reduce' do
    let(:data) { [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] }
    
    it 'raises NotImplementedError' do
      expect {
        described_class.pca_reduce(data, 2)
      }.to raise_error(NotImplementedError, /PCA reduction requires the SVD module/)
    end
    
    it 'provides helpful error message' do
      error = nil
      begin
        described_class.pca_reduce(data, 2)
      rescue NotImplementedError => e
        error = e
      end
      
      expect(error).not_to be_nil
      expect(error.message).to include('SVD module')
      expect(error.message).to include('called directly')
    end
  end
  
  describe 'module structure' do
    it 'is a module' do
      expect(described_class).to be_a(Module)
    end
    
    it 'has singleton methods' do
      expect(described_class).to respond_to(:normalize)
      expect(described_class).to respond_to(:pca_reduce)
    end
    
    it 'does not expose private methods' do
      expect(described_class).not_to respond_to(:standard_normalize)
      expect(described_class).not_to respond_to(:minmax_normalize)
      expect(described_class).not_to respond_to(:l2_normalize)
    end
  end
  
  describe 'numerical stability' do
    it 'handles identical values in all columns' do
      identical_data = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
      
      result_standard = described_class.normalize(identical_data, method: :standard)
      expect(result_standard.flatten.all? { |v| v == 0.0 }).to be true
      
      result_minmax = described_class.normalize(identical_data, method: :minmax)
      expect(result_minmax.flatten.all? { |v| v == 0.0 }).to be true
    end
    
    it 'maintains precision for reasonable values' do
      precise_data = [[1.23456789, 2.34567890], [3.45678901, 4.56789012]]
      result = described_class.normalize(precise_data, method: :standard)
      
      # Should maintain reasonable precision
      expect(result[0][0]).to be_a(Float)
      expect(result[0][0].to_s.length).to be > 3  # Has decimal places
    end
  end
end