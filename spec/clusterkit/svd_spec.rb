# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/dimensionality/svd'

RSpec.describe ClusterKit::Dimensionality::SVD do
  # Sample test matrices
  let(:simple_matrix) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }  # 3x2 matrix
  let(:square_matrix) { [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] }  # 3x3 matrix
  let(:wide_matrix) { [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] }  # 2x4 matrix
  let(:identity_matrix) { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
  let(:empty_matrix) { [] }
  let(:single_row) { [[1.0, 2.0, 3.0]] }
  let(:single_column) { [[1.0], [2.0], [3.0]] }
  
  describe '#initialize' do
    it 'creates a new SVD instance with default parameters' do
      svd = described_class.new
      expect(svd.n_components).to be_nil
      expect(svd.n_iter).to eq(2)
      expect(svd.random_seed).to be_nil
    end
    
    it 'accepts n_components parameter' do
      svd = described_class.new(n_components: 3)
      expect(svd.n_components).to eq(3)
    end
    
    it 'accepts n_iter parameter' do
      svd = described_class.new(n_iter: 5)
      expect(svd.n_iter).to eq(5)
    end
    
    it 'accepts random_seed parameter' do
      svd = described_class.new(random_seed: 42)
      expect(svd.random_seed).to eq(42)
    end
  end
  
  describe '#fit_transform' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'performs SVD and returns U, S, Vt' do
      result = svd.fit_transform(simple_matrix)
      
      expect(result).to be_a(Array)
      expect(result.size).to eq(3)
      
      u, s, vt = result
      expect(u).to be_a(Array)
      expect(s).to be_a(Array)
      expect(vt).to be_a(Array)
      
      # Check dimensions
      expect(u.size).to eq(3)  # 3 rows
      expect(u.first.size).to eq(2)  # 2 components
      expect(s.size).to eq(2)  # 2 singular values
      expect(vt.size).to eq(2)  # 2 components
      expect(vt.first.size).to eq(2)  # 2 columns
    end
    
    it 'marks the model as fitted' do
      expect(svd).not_to be_fitted
      svd.fit_transform(simple_matrix)
      expect(svd).to be_fitted
    end
    
    it 'stores the results internally' do
      u, s, vt = svd.fit_transform(simple_matrix)
      
      expect(svd.u).to eq(u)
      expect(svd.s).to eq(s)
      expect(svd.vt).to eq(vt)
    end
    
    it 'auto-determines n_components when not specified' do
      svd_auto = described_class.new
      u, s, vt = svd_auto.fit_transform(simple_matrix)
      
      # Should use min(rows, cols) = min(3, 2) = 2
      expect(s.size).to eq(2)
    end
    
    it 'handles square matrices' do
      svd_square = described_class.new(n_components: 2)
      u, s, vt = svd_square.fit_transform(square_matrix)
      
      expect(u.size).to eq(3)
      expect(s.size).to eq(2)
      expect(vt.size).to eq(2)
    end
    
    it 'handles wide matrices' do
      svd_wide = described_class.new(n_components: 2)
      u, s, vt = svd_wide.fit_transform(wide_matrix)
      
      expect(u.size).to eq(2)
      expect(s.size).to eq(2)
      expect(vt.size).to eq(2)
    end
    
    it 'validates input data' do
      expect { svd.fit_transform(nil) }.to raise_error(ArgumentError, /must be an array/)
      expect { svd.fit_transform("not an array") }.to raise_error(ArgumentError, /must be an array/)
      expect { svd.fit_transform([]) }.to raise_error(ArgumentError, /cannot be empty/)
      expect { svd.fit_transform([1, 2, 3]) }.to raise_error(ArgumentError, /must be a 2D array/)
    end
  end
  
  describe '#fit' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'fits the model and returns self' do
      result = svd.fit(simple_matrix)
      expect(result).to eq(svd)
      expect(svd).to be_fitted
    end
  end
  
  describe '#fitted?' do
    let(:svd) { described_class.new }
    
    it 'returns false before fitting' do
      expect(svd).not_to be_fitted
    end
    
    it 'returns true after fitting' do
      svd.fit(simple_matrix)
      expect(svd).to be_fitted
    end
  end
  
  describe '#components_u' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'raises error before fitting' do
      expect { svd.components_u }.to raise_error(RuntimeError, /must be fitted first/)
    end
    
    it 'returns U matrix after fitting' do
      u, _, _ = svd.fit_transform(simple_matrix)
      expect(svd.components_u).to eq(u)
    end
  end
  
  describe '#singular_values' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'raises error before fitting' do
      expect { svd.singular_values }.to raise_error(RuntimeError, /must be fitted first/)
    end
    
    it 'returns singular values after fitting' do
      _, s, _ = svd.fit_transform(simple_matrix)
      expect(svd.singular_values).to eq(s)
    end
  end
  
  describe '#components_vt' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'raises error before fitting' do
      expect { svd.components_vt }.to raise_error(RuntimeError, /must be fitted first/)
    end
    
    it 'returns Vt matrix after fitting' do
      _, _, vt = svd.fit_transform(simple_matrix)
      expect(svd.components_vt).to eq(vt)
    end
  end
  
  describe '#transform' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'raises error before fitting' do
      expect { svd.transform(simple_matrix) }.to raise_error(RuntimeError, /must be fitted first/)
    end
    
    it 'transforms fitted data' do
      svd.fit(simple_matrix)
      # Transform on same data should work
      result = svd.transform(simple_matrix)
      expect(result).to be_a(Array)
      expect(result.size).to eq(3)
      expect(result.first.size).to eq(2)
    end
    
    it 'raises NotImplementedError for new data' do
      svd.fit(simple_matrix)
      new_data = [[7.0, 8.0]]
      expect { svd.transform(new_data) }.to raise_error(NotImplementedError, /Transform for new data not yet implemented/)
    end
  end
  
  describe '#inverse_transform' do
    let(:svd) { described_class.new(n_components: 2) }
    
    it 'raises error before fitting' do
      expect { svd.inverse_transform([[1, 2]]) }.to raise_error(RuntimeError, /must be fitted first/)
    end
    
    it 'reconstructs data from transformed representation' do
      svd.fit(simple_matrix)
      transformed = svd.transform(simple_matrix)
      reconstructed = svd.inverse_transform(transformed)
      
      expect(reconstructed).to be_a(Array)
      expect(reconstructed.size).to eq(3)
      expect(reconstructed.first.size).to eq(2)
      
      # Check reconstruction is approximately correct
      # Due to truncation, it won't be exact
      simple_matrix.each_with_index do |row, i|
        row.each_with_index do |val, j|
          expect(reconstructed[i][j]).to be_within(0.1).of(val)
        end
      end
    end
  end
  
  describe '.randomized_svd' do
    it 'is a class method that performs SVD' do
      expect(described_class).to respond_to(:randomized_svd)
    end
    
    it 'delegates to the Rust implementation' do
      # The Rust implementation is in ::ClusterKit::SVD module
      expect(::ClusterKit::SVD).to respond_to(:randomized_svd_rust)
      
      u, s, vt = described_class.randomized_svd(simple_matrix, 2)
      
      expect(u).to be_a(Array)
      expect(s).to be_a(Array) 
      expect(vt).to be_a(Array)
    end
    
    it 'accepts n_iter parameter' do
      u, s, vt = described_class.randomized_svd(simple_matrix, 2, n_iter: 5)
      
      expect(u).to be_a(Array)
      expect(s).to be_a(Array)
      expect(vt).to be_a(Array)
    end
    
    it 'handles identity matrix' do
      u, s, vt = described_class.randomized_svd(identity_matrix, 3)
      
      # Singular values of identity matrix should be all 1s
      expect(s).to all(be_within(1e-10).of(1.0))
    end
    
    it 'validates k parameter' do
      expect {
        described_class.randomized_svd(simple_matrix, 10)
      }.to raise_error(ArgumentError, /cannot be larger than/)
    end
  end
  
  describe 'integration' do
    it 'works with the module-level convenience method' do
      u, s, vt = ClusterKit.svd(simple_matrix, 2)
      
      expect(u).to be_a(Array)
      expect(s).to be_a(Array)
      expect(vt).to be_a(Array)
    end
    
    it 'can be used for dimensionality reduction' do
      # Generate high-dimensional data
      high_dim = 20.times.map { 10.times.map { rand } }
      
      svd = described_class.new(n_components: 3)
      u, s, vt = svd.fit_transform(high_dim)
      
      # U * S gives the reduced representation
      reduced = u.map.with_index do |row, i|
        row.map.with_index { |val, j| val * s[j] }
      end
      
      expect(reduced.size).to eq(20)  # Same number of samples
      expect(reduced.first.size).to eq(3)  # Reduced to 3 dimensions
    end
  end
  
  describe 'API consistency' do
    let(:svd) { described_class.new(n_components: 2) }
    let(:pca) { ClusterKit::Dimensionality::PCA.new(n_components: 2) }
    
    it 'has similar initialization pattern to PCA' do
      expect(svd).to respond_to(:fit)
      expect(svd).to respond_to(:fit_transform)
      expect(svd).to respond_to(:fitted?)
      
      expect(pca).to respond_to(:fit)
      expect(pca).to respond_to(:fit_transform)
      expect(pca).to respond_to(:fitted?)
    end
    
    it 'returns self from fit for method chaining' do
      expect(svd.fit(simple_matrix)).to eq(svd)
      expect(pca.fit(simple_matrix)).to eq(pca)
    end
  end
end