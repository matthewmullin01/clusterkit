# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'clusterkit extension loading' do
  it 'loads the native extension' do
    # The extension should already be loaded by the time tests run
    # Verify that the private RustUMAP constant exists (even though it's private)
    expect(ClusterKit.const_defined?(:RustUMAP, false)).to be true
    # And verify public API is available
    expect(defined?(ClusterKit::Dimensionality::UMAP)).to eq('constant')
  end
  
  it 'defines Utils methods on the module' do
    expect(ClusterKit::Utils).to respond_to(:estimate_intrinsic_dimension_rust)
    expect(ClusterKit::Utils).to respond_to(:estimate_hubness_rust)
  end
  
  it 'defines SVD methods on the module' do
    expect(ClusterKit::SVD).to respond_to(:randomized_svd_rust)
  end
  
  context 'extension loading mechanism' do
    # This is a bit tricky to test since the file is already loaded
    # We can at least document the loading behavior
    it 'attempts to load .bundle first on macOS' do
      # On macOS, the extension is compiled as .bundle
      # This test documents that behavior
      # The extension was renamed from annembed_ruby to clusterkit
      expect(File.extname($LOADED_FEATURES.grep(/clusterkit\.(bundle|so)/).first || '')).to match(/\.(bundle|so)/)
    end
    
    it 'has a fallback mechanism for different platforms' do
      # The loading code has a rescue clause for LoadError
      # This ensures compatibility across macOS (.bundle) and Linux (.so)
      # We can't easily test the fallback without mocking require,
      # but we can verify the extension loaded successfully
      # Verify extension loaded by checking public API
      expect(ClusterKit::Dimensionality::UMAP).to be_a(Class)
    end
  end
  
  describe 'UMAP public API' do
    it 'exists and can be instantiated through Dimensionality module' do
      # The public API is through Dimensionality::UMAP
      umap = ClusterKit::Dimensionality::UMAP.new(n_neighbors: 5, n_components: 2)
      expect(umap).to be_a(ClusterKit::Dimensionality::UMAP)
      
      # fit_transform actually works with proper data
      test_data = Array.new(20) { Array.new(5) { rand } }
      result = umap.fit_transform(test_data)
      expect(result).to be_a(Array)
      expect(result.size).to eq(20)
      expect(result.first.size).to eq(2)
    end
    
    it 'responds to expected methods' do
      # Public API methods
      expect(ClusterKit::Dimensionality::UMAP.instance_methods).to include(:fit, :fit_transform, :transform, :save_model, :fitted?)
      expect(ClusterKit::Dimensionality::UMAP.singleton_methods).to include(:load_model, :save_data, :load_data)
    end
  end
end