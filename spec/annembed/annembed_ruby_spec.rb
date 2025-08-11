# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'annembed_ruby extension loading' do
  it 'loads the native extension' do
    # The extension should already be loaded by the time tests run
    # We just verify that the expected modules and methods exist
    expect(defined?(AnnEmbed::RustUMAP)).to eq('constant')
    expect(AnnEmbed::RustUMAP).to respond_to(:new)
  end
  
  it 'defines Utils methods on the module' do
    expect(AnnEmbed::Utils).to respond_to(:estimate_intrinsic_dimension_rust)
    expect(AnnEmbed::Utils).to respond_to(:estimate_hubness_rust)
  end
  
  it 'defines SVD methods on the module' do
    expect(AnnEmbed::SVD).to respond_to(:randomized_svd_rust)
  end
  
  context 'extension loading mechanism' do
    # This is a bit tricky to test since the file is already loaded
    # We can at least document the loading behavior
    it 'attempts to load .bundle first on macOS' do
      # On macOS, the extension is compiled as .bundle
      # This test documents that behavior
      expect(File.extname($LOADED_FEATURES.grep(/annembed_ruby/).first || '')).to match(/\.(bundle|so)/)
    end
    
    it 'has a fallback mechanism for different platforms' do
      # The loading code has a rescue clause for LoadError
      # This ensures compatibility across macOS (.bundle) and Linux (.so)
      # We can't easily test the fallback without mocking require,
      # but we can verify the extension loaded successfully
      expect(AnnEmbed::RustUMAP).to be_a(Class)
    end
  end
  
  describe 'RustUMAP class' do
    it 'exists and can be instantiated' do
      # RustUMAP.new with proper parameters creates an instance
      config = { n_neighbors: 5, n_components: 2 }
      umap = AnnEmbed::RustUMAP.new(config)
      expect(umap).to be_a(AnnEmbed::RustUMAP)
      
      # fit_transform actually works now with proper data
      test_data = Array.new(20) { Array.new(5) { rand } }
      result = umap.fit_transform(test_data)
      expect(result).to be_a(Array)
      expect(result.size).to eq(20)
      expect(result.first.size).to eq(2)
    end
    
    it 'responds to expected methods' do
      # These methods throw NotImplementedError but should exist
      expect(AnnEmbed::RustUMAP.instance_methods).to include(:fit_transform, :transform, :save_model)
      expect(AnnEmbed::RustUMAP.singleton_methods).to include(:load_model)
    end
  end
end