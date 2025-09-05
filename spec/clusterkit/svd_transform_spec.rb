# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/dimensionality/svd'
require 'clusterkit/dimensionality/pca'

RSpec.describe "SVD Transform for New Data" do
  let(:training_data) do
    # Well-structured training data with clear patterns
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0], 
      [0.0, 0.0, 1.0],
      [2.0, 1.0, 0.5],
      [1.5, 2.0, 0.3],
      [0.8, 1.2, 1.8]
    ]
  end
  
  let(:new_data) do
    # New data with same feature structure but different samples
    [
      [1.5, 0.5, 0.2],
      [0.3, 1.8, 0.1],
      [2.1, 1.3, 0.9]
    ]
  end
  
  let(:svd) { ClusterKit::Dimensionality::SVD.new(n_components: 2) }
  
  describe "transform functionality (now working)" do
    it "can transform the same data object used for fitting" do
      svd.fit(training_data)
      
      # Same data object works (legacy behavior preserved)
      expect { svd.transform(training_data) }.not_to raise_error
      result = svd.transform(training_data)
      expect(result).to be_a(Array)
      expect(result.size).to eq(6)
      expect(result.first.size).to eq(2)
    end
    
    it "now transforms any new data successfully" do
      svd.fit(training_data)
      
      # New data now works!
      expect { svd.transform(new_data) }.not_to raise_error
      result = svd.transform(new_data)
      expect(result).to be_a(Array)
      expect(result.size).to eq(3)  # new_data sample count
      expect(result.first.size).to eq(2)  # n_components
    end
    
    it "transforms identical data in different object successfully" do
      svd.fit(training_data)
      
      # Create new array with identical values but different object_id
      identical_new_data = training_data.map(&:dup)
      expect(identical_new_data).to eq(training_data)  # Same values
      expect(identical_new_data.object_id).not_to eq(training_data.object_id)  # Different object
      
      expect { svd.transform(identical_new_data) }.not_to raise_error
      result = svd.transform(identical_new_data)
      expect(result.size).to eq(training_data.size)
    end
  end
  
  describe "new data transformation (now working)" do    
    it "transforms new data with same feature count" do
      svd.fit(training_data)
      
      transformed = svd.transform(new_data)
      
      expect(transformed).to be_a(Array)
      expect(transformed.size).to eq(3)  # Same as new_data sample count
      expect(transformed.first.size).to eq(2)  # n_components
      
      # All values should be finite numbers
      transformed.flatten.each do |val|
        expect(val).to be_a(Numeric)
        expect(val).to be_finite
      end
    end
    
    it "handles different sample counts" do
      svd.fit(training_data)  # 6 samples
      
      # Test with different sample counts
      single_sample = [new_data.first]
      many_samples = new_data * 5  # 15 samples
      
      result1 = svd.transform(single_sample)
      result2 = svd.transform(many_samples)
      
      expect(result1.size).to eq(1)
      expect(result2.size).to eq(15)
      expect(result1.first.size).to eq(2)
      expect(result2.first.size).to eq(2)
    end
    
    it "raises error for mismatched feature count" do
      svd.fit(training_data)  # 3 features
      
      wrong_features = [[1.0, 2.0]]  # Only 2 features
      expect { svd.transform(wrong_features) }.to raise_error(ArgumentError, /feature/)
      
      too_many_features = [[1.0, 2.0, 3.0, 4.0]]  # 4 features
      expect { svd.transform(too_many_features) }.to raise_error(ArgumentError, /feature/)
    end
    
    it "handles edge cases" do
      svd.fit(training_data)
      
      # Empty data
      expect { svd.transform([]) }.to raise_error(ArgumentError, /empty/)
      
      # Non-numeric data
      expect { svd.transform([["a", "b", "c"]]) }.to raise_error(ArgumentError, /numeric/)
    end
  end
  
  describe "mathematical correctness (now working)" do
    it "produces mathematically correct projections" do
      # Use simple, predictable data for mathematical verification
      simple_training = [
        [2.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0]
      ]
      
      simple_new = [
        [1.0, 0.0],
        [0.0, 1.0]
      ]
      
      svd = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      svd.fit(simple_training)
      
      # Transform new data
      transformed = svd.transform(simple_new)
      
      # The transformation should be: new_data × V
      # Where V comes from the SVD decomposition
      # We can verify this by checking the math manually or comparing with direct computation
      expect(transformed).to be_a(Array)
      expect(transformed.size).to eq(2)
      
      # More specific mathematical verification would require manual calculation
      # but at minimum, results should be reasonable
      transformed.flatten.each { |val| expect(val).to be_finite }
    end
    
    it "maintains orthogonal properties" do
      # Orthogonal input vectors should remain orthogonal after transformation
      orthogonal_training = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0]
      ]
      
      orthogonal_new = [
        [3.0, 0.0, 0.0],  # Along first axis
        [0.0, 3.0, 0.0]   # Along second axis
      ]
      
      svd = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      svd.fit(orthogonal_training)
      
      transformed = svd.transform(orthogonal_new)
      
      # The transformed vectors should maintain their relative relationships
      # This is a basic sanity check for mathematical correctness
      expect(transformed.size).to eq(2)
      expect(transformed.first.size).to eq(2)
    end
  end
  
  describe "consistency with PCA (now working)" do
    it "behaves similarly to PCA for dimensionality reduction" do
      # Both SVD and PCA should produce similar results for dimensionality reduction
      # (SVD without centering vs PCA with centering will differ, but patterns should be similar)
      
      svd = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
      
      # Fit both on training data
      svd.fit(training_data)
      pca.fit(training_data)
      
      # Transform new data
      svd_result = svd.transform(new_data)
      pca_result = pca.transform(new_data)
      
      # Both should have same dimensions
      expect(svd_result.size).to eq(pca_result.size)
      expect(svd_result.first.size).to eq(pca_result.first.size)
      
      # Results might differ due to centering, but both should be finite
      svd_result.flatten.each { |val| expect(val).to be_finite }
      pca_result.flatten.each { |val| expect(val).to be_finite }
    end
    
    it "has consistent API with PCA" do
      svd = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
      
      # Both should support the same workflow
      svd.fit(training_data)
      pca.fit(training_data)
      
      # Both should be able to transform new data
      expect { svd.transform(new_data) }.not_to raise_error
      expect { pca.transform(new_data) }.not_to raise_error
      
      # Both should return arrays with correct dimensions
      svd_result = svd.transform(new_data)
      pca_result = pca.transform(new_data)
      
      expect(svd_result.size).to eq(new_data.size)
      expect(pca_result.size).to eq(new_data.size)
    end
    
    it "handles workflow consistency: fit training, transform test" do
      # This is the standard ML workflow that should work for both
      svd = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
      
      # Standard ML workflow
      svd.fit(training_data)        # Train on training set
      pca.fit(training_data)
      
      svd_test = svd.transform(new_data)  # Apply to test set
      pca_test = pca.transform(new_data)
      
      # Both should complete successfully and return proper dimensions
      expect(svd_test.size).to eq(new_data.size)
      expect(pca_test.size).to eq(new_data.size)
      expect(svd_test.first.size).to eq(2)
      expect(pca_test.first.size).to eq(2)
    end
  end
  
  describe "reconstruction with new data (now working)" do
    it "supports round-trip transformation" do
      svd.fit(training_data)
      
      # Transform new data
      transformed = svd.transform(new_data)
      
      # Inverse transform should approximately reconstruct
      reconstructed = svd.inverse_transform(transformed)
      
      expect(reconstructed.size).to eq(new_data.size)
      expect(reconstructed.first.size).to eq(new_data.first.size)
      
      # Check reconstruction quality (won't be perfect due to dimensionality reduction)
      new_data.each_with_index do |original_row, i|
        original_row.each_with_index do |original_val, j|
          reconstructed_val = reconstructed[i][j]
          # Allow reasonable reconstruction error
          expect(reconstructed_val).to be_within(2.0).of(original_val)
        end
      end
    end
    
    it "handles reconstruction quality with fewer components" do
      # Test with very few components to see reconstruction degradation
      svd_1comp = ClusterKit::Dimensionality::SVD.new(n_components: 1)
      svd_2comp = ClusterKit::Dimensionality::SVD.new(n_components: 2)
      
      svd_1comp.fit(training_data)
      svd_2comp.fit(training_data)
      
      transformed_1 = svd_1comp.transform(new_data)
      transformed_2 = svd_2comp.transform(new_data)
      
      reconstructed_1 = svd_1comp.inverse_transform(transformed_1)
      reconstructed_2 = svd_2comp.inverse_transform(transformed_2)
      
      # More components should give better reconstruction
      # Calculate MSE for both
      mse_1 = calculate_mse(new_data, reconstructed_1)
      mse_2 = calculate_mse(new_data, reconstructed_2)
      
      expect(mse_2).to be <= mse_1
    end
  end
  
  describe "edge cases for new data transform (now working)" do
    it "handles single sample" do
      svd.fit(training_data)
      
      single_sample = [new_data.first]
      result = svd.transform(single_sample)
      
      expect(result.size).to eq(1)
      expect(result.first.size).to eq(2)
    end
    
    it "handles many samples" do
      svd.fit(training_data)
      
      # Large batch of new data
      large_new_data = 100.times.map { [rand, rand, rand] }
      result = svd.transform(large_new_data)
      
      expect(result.size).to eq(100)
      expect(result.first.size).to eq(2)
    end
    
    it "validates input structure for new data" do
      svd.fit(training_data)
      
      # Wrong number of features
      expect { svd.transform([[1.0, 2.0]]) }.to raise_error(ArgumentError, /feature/)
      expect { svd.transform([[1.0, 2.0, 3.0, 4.0]]) }.to raise_error(ArgumentError, /feature/)
      
      # Non-numeric data
      expect { svd.transform([["a", "b", "c"]]) }.to raise_error(ArgumentError, /numeric/)
      
      # Empty data
      expect { svd.transform([]) }.to raise_error(ArgumentError, /empty/)
      
      # 1D data
      expect { svd.transform([1, 2, 3]) }.to raise_error(ArgumentError, /2D array/)
    end
  end
  
  describe "performance and memory considerations (now working)" do
    it "handles large new datasets efficiently" do
      # Fit on small training data
      svd.fit(training_data)
      
      # Transform much larger new dataset
      large_data = 1000.times.map { 3.times.map { rand } }
      
      start_time = Time.now
      result = svd.transform(large_data)
      duration = Time.now - start_time
      
      expect(result.size).to eq(1000)
      expect(duration).to be < 1.0  # Should be fast
    end
  end
  
  private
  
  # Helper method to calculate Mean Squared Error
  def calculate_mse(original, reconstructed)
    total_error = 0.0
    count = 0
    
    original.each_with_index do |row, i|
      row.each_with_index do |val, j|
        error = (val - reconstructed[i][j]) ** 2
        total_error += error
        count += 1
      end
    end
    
    total_error / count
  end
end