# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit::Config do
  describe '#initialize' do
    context 'with default parameters' do
      subject(:config) { described_class.new }
      
      it 'defaults to UMAP method' do
        expect(config.method).to eq(:umap)
      end
      
      it 'sets common defaults' do
        expect(config.n_components).to eq(2)
        expect(config.random_seed).to be_nil
        expect(config.n_threads).to be_nil
      end
      
      it 'sets HNSW defaults' do
        expect(config.ef_construction).to eq(200)
        expect(config.max_nb_connection).to eq(16)
        expect(config.nb_layer).to eq(16)
      end
      
      it 'sets UMAP-specific defaults' do
        expect(config.n_neighbors).to eq(15)
        expect(config.min_dist).to eq(0.1)
        expect(config.spread).to eq(1.0)
        expect(config.local_connectivity).to eq(1.0)
        expect(config.set_op_mix_ratio).to eq(1.0)
        expect(config.negative_sample_rate).to eq(5)
        expect(config.transform_queue_size).to eq(4.0)
      end
    end
    
    context 'with t-SNE method' do
      subject(:config) { described_class.new(method: :tsne) }
      
      it 'sets method to tsne' do
        expect(config.method).to eq(:tsne)
      end
      
      it 'sets t-SNE specific defaults' do
        expect(config.perplexity).to eq(30.0)
        expect(config.learning_rate).to eq(200.0)
        expect(config.n_iter).to eq(1000)
        expect(config.early_exaggeration).to eq(12.0)
        expect(config.theta).to eq(0.5)
        expect(config.eta).to eq(200.0)
      end
      
      it 'still sets common defaults' do
        expect(config.n_components).to eq(2)
        expect(config.ef_construction).to eq(200)
      end
    end
    
    context 'with LargeVis method' do
      subject(:config) { described_class.new(method: :largevis) }
      
      it 'sets method to largevis' do
        expect(config.method).to eq(:largevis)
      end
      
      it 'sets LargeVis specific defaults' do
        expect(config.n_neighbors).to eq(15)
        expect(config.perplexity).to eq(30.0)
        expect(config.learning_rate).to eq(1.0)
        expect(config.n_iter).to eq(1000)
      end
    end
    
    context 'with diffusion method' do
      subject(:config) { described_class.new(method: :diffusion) }
      
      it 'sets method to diffusion' do
        expect(config.method).to eq(:diffusion)
      end
      
      it 'sets diffusion specific defaults' do
        expect(config.n_neighbors).to eq(15)
        # Note: alpha is set but not exposed as attr_accessor in the current implementation
        expect(config.n_iter).to eq(1)
      end
    end
    
    context 'with custom options' do
      subject(:config) do
        described_class.new(
          method: :umap,
          n_components: 3,
          n_neighbors: 30,
          random_seed: 42,
          min_dist: 0.2
        )
      end
      
      it 'overrides defaults with provided options' do
        expect(config.n_components).to eq(3)
        expect(config.n_neighbors).to eq(30)
        expect(config.random_seed).to eq(42)
        expect(config.min_dist).to eq(0.2)
      end
      
      it 'keeps other defaults unchanged' do
        expect(config.spread).to eq(1.0)
        expect(config.local_connectivity).to eq(1.0)
      end
    end
    
    context 'with unknown options' do
      it 'warns about unknown options' do
        expect {
          described_class.new(unknown_option: 'value')
        }.to output(/Unknown option: unknown_option/).to_stderr
      end
      
      it 'still creates the config object' do
        config = nil
        expect {
          config = described_class.new(unknown_option: 'value')
        }.to output.to_stderr
        
        expect(config).to be_a(described_class)
        expect(config.method).to eq(:umap)
      end
    end
  end
  
  describe '#to_h' do
    subject(:config) do
      described_class.new(
        method: :umap,
        n_components: 3,
        n_neighbors: 20,
        random_seed: 123
      )
    end
    
    it 'converts config to hash' do
      hash = config.to_h
      expect(hash).to be_a(Hash)
      expect(hash[:method]).to eq(:umap)
      expect(hash[:n_components]).to eq(3)
      expect(hash[:n_neighbors]).to eq(20)
      expect(hash[:random_seed]).to eq(123)
    end
    
    it 'includes all instance variables' do
      hash = config.to_h
      # Check that common parameters are included
      expect(hash).to include(
        :method,
        :n_components,
        :n_neighbors,
        :random_seed,
        :min_dist,
        :spread,
        :ef_construction,
        :max_nb_connection
      )
    end
    
    it 'excludes @ prefix from instance variable names' do
      hash = config.to_h
      hash.keys.each do |key|
        expect(key.to_s).not_to start_with('@')
      end
    end
  end
  
  describe 'attribute accessors' do
    subject(:config) { described_class.new }
    
    it 'allows reading and writing HNSW parameters' do
      config.ef_construction = 300
      config.max_nb_connection = 32
      config.nb_layer = 8
      
      expect(config.ef_construction).to eq(300)
      expect(config.max_nb_connection).to eq(32)
      expect(config.nb_layer).to eq(8)
    end
    
    it 'allows reading and writing common parameters' do
      config.n_components = 5
      config.n_neighbors = 25
      config.random_seed = 999
      config.n_threads = 4
      
      expect(config.n_components).to eq(5)
      expect(config.n_neighbors).to eq(25)
      expect(config.random_seed).to eq(999)
      expect(config.n_threads).to eq(4)
    end
    
    it 'allows reading and writing UMAP parameters' do
      config.min_dist = 0.5
      config.spread = 2.0
      config.local_connectivity = 2.0
      config.set_op_mix_ratio = 0.5
      config.negative_sample_rate = 10
      config.transform_queue_size = 8.0
      
      expect(config.min_dist).to eq(0.5)
      expect(config.spread).to eq(2.0)
      expect(config.local_connectivity).to eq(2.0)
      expect(config.set_op_mix_ratio).to eq(0.5)
      expect(config.negative_sample_rate).to eq(10)
      expect(config.transform_queue_size).to eq(8.0)
    end
    
    it 'allows reading and writing t-SNE parameters' do
      config.perplexity = 50.0
      config.learning_rate = 100.0
      config.n_iter = 2000
      config.early_exaggeration = 24.0
      config.theta = 0.3
      config.eta = 150.0
      
      expect(config.perplexity).to eq(50.0)
      expect(config.learning_rate).to eq(100.0)
      expect(config.n_iter).to eq(2000)
      expect(config.early_exaggeration).to eq(24.0)
      expect(config.theta).to eq(0.3)
      expect(config.eta).to eq(150.0)
    end
  end
  
  describe 'parameter preservation' do
    it 'preserves parameters across method changes' do
      config = described_class.new(method: :umap, n_components: 5)
      expect(config.n_components).to eq(5)
      
      # Manually changing method shouldn't reset other parameters
      config.method = :tsne
      expect(config.n_components).to eq(5)
    end
    
    it 'allows nil values for optional parameters' do
      config = described_class.new(random_seed: nil, n_threads: nil)
      expect(config.random_seed).to be_nil
      expect(config.n_threads).to be_nil
    end
  end
  
  describe 'edge cases' do
    it 'handles empty initialization' do
      config = described_class.new
      expect(config).to be_a(described_class)
      expect(config.method).to eq(:umap)
    end
    
    it 'handles symbol and string method names' do
      config1 = described_class.new(method: :tsne)
      config2 = described_class.new(method: 'tsne')
      
      expect(config1.method).to eq(:tsne)
      expect(config2.method).to eq('tsne')
    end
    
    it 'allows updating parameters after initialization' do
      config = described_class.new
      original_neighbors = config.n_neighbors
      
      config.n_neighbors = original_neighbors + 10
      expect(config.n_neighbors).to eq(original_neighbors + 10)
    end
  end
  
  describe 'integration with to_h' do
    it 'round-trips parameters through to_h' do
      original_config = described_class.new(
        method: :umap,
        n_components: 4,
        n_neighbors: 25,
        min_dist: 0.3
      )
      
      hash = original_config.to_h
      new_config = described_class.new(**hash)
      
      expect(new_config.method).to eq(original_config.method)
      expect(new_config.n_components).to eq(original_config.n_components)
      expect(new_config.n_neighbors).to eq(original_config.n_neighbors)
      expect(new_config.min_dist).to eq(original_config.min_dist)
    end
  end
end