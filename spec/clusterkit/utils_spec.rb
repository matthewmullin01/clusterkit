# frozen_string_literal: true

require 'spec_helper'
require 'annembed/utils'

RSpec.describe ClusterKit::Utils do
  # Sample test data
  let(:simple_data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
  let(:high_dim_data) { Array.new(20) { Array.new(10) { rand } } }
  let(:low_dim_data) { Array.new(20) { Array.new(2) { rand } } }
  let(:empty_data) { [] }
  
  describe '.estimate_intrinsic_dimension' do
    context 'with valid data' do
      it 'estimates dimension for simple data' do
        # Mock the rust method since it's not fully implemented yet
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(2.5)
        
        result = described_class.estimate_intrinsic_dimension(simple_data)
        expect(result).to eq(2.5)
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension_rust).with(simple_data, 10)
      end
      
      it 'accepts custom k_neighbors parameter' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(3.2)
        
        result = described_class.estimate_intrinsic_dimension(simple_data, k_neighbors: 5)
        expect(result).to eq(3.2)
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension_rust).with(simple_data, 5)
      end
      
      it 'works with high-dimensional data' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(7.8)
        
        result = described_class.estimate_intrinsic_dimension(high_dim_data, k_neighbors: 15)
        expect(result).to be_a(Float)
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension_rust).with(high_dim_data, 15)
      end
      
      it 'currently raises NotImplementedError from Rust' do
        # This documents the current state - the Rust method is not implemented yet
        expect {
          described_class.estimate_intrinsic_dimension(high_dim_data)
        }.to raise_error(NotImplementedError, /Dimension estimation not implemented yet/)
      end
    end
    
    context 'with invalid data' do
      it 'raises error for non-array data' do
        expect {
          described_class.estimate_intrinsic_dimension("not an array")
        }.to raise_error(ArgumentError, /Unsupported data type: String/)
      end
      
      it 'raises error for nil data' do
        expect {
          described_class.estimate_intrinsic_dimension(nil)
        }.to raise_error(ArgumentError, /Unsupported data type: NilClass/)
      end
      
      it 'raises error for hash data' do
        expect {
          described_class.estimate_intrinsic_dimension({ invalid: 'data' })
        }.to raise_error(ArgumentError, /Unsupported data type: Hash/)
      end
    end
    
    context 'edge cases' do
      it 'handles empty data gracefully' do
        # The Rust implementation should handle this
        expect {
          described_class.estimate_intrinsic_dimension(empty_data)
        }.to raise_error(NotImplementedError) # Currently not implemented
      end
      
      it 'handles single point data' do
        single_point = [[1.0, 2.0, 3.0]]
        expect {
          described_class.estimate_intrinsic_dimension(single_point)
        }.to raise_error(NotImplementedError) # Currently not implemented
      end
      
      it 'handles small k_neighbors' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(1.5)
        
        result = described_class.estimate_intrinsic_dimension(simple_data, k_neighbors: 1)
        expect(result).to eq(1.5)
      end
      
      it 'handles large k_neighbors' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(4.2)
        
        result = described_class.estimate_intrinsic_dimension(high_dim_data, k_neighbors: 50)
        expect(result).to eq(4.2)
      end
    end
  end
  
  describe '.estimate_hubness' do
    context 'with valid data' do
      it 'estimates hubness for simple data' do
        # Mock the rust method
        mock_result = {
          'mean_hub_score' => 0.5,
          'max_hub_score' => 0.9,
          'hubness_skewness' => 0.2
        }
        allow(ClusterKit::Utils).to receive(:estimate_hubness_rust).and_return(mock_result)
        
        result = described_class.estimate_hubness(simple_data)
        
        expect(result).to be_a(Hash)
        expect(result[:mean_hub_score]).to eq(0.5)
        expect(result[:max_hub_score]).to eq(0.9)
        expect(result[:hubness_skewness]).to eq(0.2)
        expect(ClusterKit::Utils).to have_received(:estimate_hubness_rust).with(simple_data)
      end
      
      it 'symbolizes hash keys from Rust' do
        mock_result = {
          'mean_hub_score' => 0.5,
          'std_hub_score' => 0.1,
          'max_hub_score' => 0.9
        }
        allow(ClusterKit::Utils).to receive(:estimate_hubness_rust).and_return(mock_result)
        
        result = described_class.estimate_hubness(simple_data)
        
        expect(result.keys).to all(be_a(Symbol))
        expect(result.keys).to include(:mean_hub_score, :std_hub_score, :max_hub_score)
      end
      
      it 'currently raises NotImplementedError from Rust' do
        # This documents the current state - the Rust method is not implemented yet
        expect {
          described_class.estimate_hubness(high_dim_data)
        }.to raise_error(NotImplementedError)
      end
    end
    
    context 'with invalid data' do
      it 'raises error for non-array data' do
        expect {
          described_class.estimate_hubness("not an array")
        }.to raise_error(ArgumentError, /Unsupported data type: String/)
      end
      
      it 'raises error for nil data' do
        expect {
          described_class.estimate_hubness(nil)
        }.to raise_error(ArgumentError, /Unsupported data type: NilClass/)
      end
      
      it 'raises error for numeric data' do
        expect {
          described_class.estimate_hubness(123)
        }.to raise_error(ArgumentError, /Unsupported data type: Integer/)
      end
    end
    
    context 'edge cases' do
      it 'handles empty data' do
        expect {
          described_class.estimate_hubness(empty_data)
        }.to raise_error(NotImplementedError) # Currently not implemented
      end
      
      it 'handles single point data' do
        single_point = [[1.0, 2.0, 3.0]]
        expect {
          described_class.estimate_hubness(single_point)
        }.to raise_error(NotImplementedError) # Currently not implemented
      end
      
      it 'handles identical points' do
        identical_data = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
        # This might work but return special values
        allow(ClusterKit::Utils).to receive(:estimate_hubness_rust).and_return({ 'mean_hub_score' => 0.0 })
        
        result = described_class.estimate_hubness(identical_data)
        expect(result[:mean_hub_score]).to eq(0.0)
      end
    end
  end
  
  describe '.neighborhood_stability' do
    let(:original_data) { high_dim_data }
    let(:embedded_data) { low_dim_data }
    
    context 'with valid data' do
      it 'raises NotImplementedError' do
        expect {
          described_class.neighborhood_stability(original_data, embedded_data)
        }.to raise_error(NotImplementedError, /Neighborhood stability not implemented yet/)
      end
      
      it 'validates original data before raising' do
        expect {
          described_class.neighborhood_stability("invalid", embedded_data)
        }.to raise_error(ArgumentError, /Unsupported data type: String/)
      end
      
      it 'validates embedded data before raising' do
        expect {
          described_class.neighborhood_stability(original_data, "invalid")
        }.to raise_error(ArgumentError, /Unsupported data type: String/)
      end
      
      it 'accepts k parameter' do
        expect {
          described_class.neighborhood_stability(original_data, embedded_data, k: 10)
        }.to raise_error(NotImplementedError)
      end
    end
    
    context 'with invalid data' do
      it 'raises error for nil original data' do
        expect {
          described_class.neighborhood_stability(nil, embedded_data)
        }.to raise_error(ArgumentError, /Unsupported data type: NilClass/)
      end
      
      it 'raises error for nil embedded data' do
        expect {
          described_class.neighborhood_stability(original_data, nil)
        }.to raise_error(ArgumentError, /Unsupported data type: NilClass/)
      end
      
      it 'raises error for hash original data' do
        expect {
          described_class.neighborhood_stability({ invalid: 'data' }, embedded_data)
        }.to raise_error(ArgumentError, /Unsupported data type: Hash/)
      end
      
      it 'raises error for hash embedded data' do
        expect {
          described_class.neighborhood_stability(original_data, { invalid: 'data' })
        }.to raise_error(ArgumentError, /Unsupported data type: Hash/)
      end
    end
  end
  
  describe 'private methods' do
    describe '#symbolize_keys' do
      it 'converts string keys to symbols' do
        hash = { 'key1' => 'value1', 'key2' => 'value2' }
        result = described_class.send(:symbolize_keys, hash)
        
        expect(result).to eq({ key1: 'value1', key2: 'value2' })
      end
      
      it 'handles empty hash' do
        result = described_class.send(:symbolize_keys, {})
        expect(result).to eq({})
      end
      
      it 'handles mixed key types' do
        hash = { 'string_key' => 1, :symbol_key => 2, 123 => 3 }
        result = described_class.send(:symbolize_keys, hash)
        
        expect(result[:string_key]).to eq(1)
        expect(result[:symbol_key]).to eq(2)
        expect(result[:"123"]).to eq(3)
      end
      
      it 'returns non-hash values unchanged' do
        expect(described_class.send(:symbolize_keys, nil)).to be_nil
        expect(described_class.send(:symbolize_keys, "string")).to eq("string")
        expect(described_class.send(:symbolize_keys, 123)).to eq(123)
        expect(described_class.send(:symbolize_keys, [])).to eq([])
      end
      
      it 'does not modify nested hashes' do
        hash = { 'outer' => { 'inner' => 'value' } }
        result = described_class.send(:symbolize_keys, hash)
        
        expect(result[:outer]).to eq({ 'inner' => 'value' })
      end
    end
  end
  
  describe 'module structure' do
    it 'is a module' do
      expect(described_class).to be_a(Module)
    end
    
    it 'has public methods' do
      expect(described_class).to respond_to(:estimate_intrinsic_dimension)
      expect(described_class).to respond_to(:estimate_hubness)
      expect(described_class).to respond_to(:neighborhood_stability)
    end
    
    it 'does not expose private methods' do
      expect(described_class).not_to respond_to(:symbolize_keys)
    end
    
    it 'exposes rust methods on the module itself' do
      # The rust methods are actually added to the Utils module directly
      expect(ClusterKit::Utils).to respond_to(:estimate_intrinsic_dimension_rust)
      expect(ClusterKit::Utils).to respond_to(:estimate_hubness_rust)
    end
  end
  
  describe 'integration tests' do
    context 'with real data flow' do
      let(:test_data) { Array.new(50) { Array.new(5) { rand * 10 } } }
      
      it 'can estimate dimension and hubness for same dataset' do
        # Mock both methods since they're not implemented yet
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).and_return(3.5)
        allow(ClusterKit::Utils).to receive(:estimate_hubness_rust).and_return({ 'mean_hub_score' => 0.5 })
        
        dimension = described_class.estimate_intrinsic_dimension(test_data, k_neighbors: 10)
        hubness = described_class.estimate_hubness(test_data)
        
        expect(dimension).to eq(3.5)
        expect(hubness).to be_a(Hash)
        expect(hubness).to have_key(:mean_hub_score)
      end
      
      it 'handles data with varying dimensions consistently' do
        # Mock to simulate expected behavior
        data_2d = Array.new(30) { Array.new(2) { rand } }
        data_10d = Array.new(30) { Array.new(10) { rand } }
        
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).with(data_2d, 5).and_return(1.8)
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension_rust).with(data_10d, 5).and_return(7.2)
        
        dim_2d = described_class.estimate_intrinsic_dimension(data_2d, k_neighbors: 5)
        dim_10d = described_class.estimate_intrinsic_dimension(data_10d, k_neighbors: 5)
        
        expect(dim_2d).to be < dim_10d
      end
    end
  end
end