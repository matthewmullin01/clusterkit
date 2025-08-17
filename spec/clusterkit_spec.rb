# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit do
  describe 'module constants' do
    it 'defines METHODS constant' do
      expect(ClusterKit::METHODS).to be_a(Array)
      expect(ClusterKit::METHODS).to include(:umap, :tsne, :largevis, :diffusion)
    end
    
    it 'has frozen METHODS array' do
      expect(ClusterKit::METHODS).to be_frozen
    end
  end
  
  describe 'error classes' do
    it 'defines Error as StandardError subclass' do
      expect(ClusterKit::Error).to be < StandardError
    end
    
    it 'defines DimensionError' do
      expect(ClusterKit::DimensionError).to be < ClusterKit::Error
    end
    
    it 'defines ConvergenceError' do
      expect(ClusterKit::ConvergenceError).to be < ClusterKit::Error
    end
    
    it 'defines InvalidParameterError' do
      expect(ClusterKit::InvalidParameterError).to be < ClusterKit::Error
    end
    
    it 'allows raising custom errors' do
      expect { raise ClusterKit::DimensionError, "test" }.to raise_error(ClusterKit::DimensionError, "test")
    end
  end
  
  describe 'autoloaded classes' do
    it 'autoloads UMAP' do
      expect(defined?(ClusterKit::UMAP)).to eq('constant')
    end
    
    it 'autoloads Config' do
      expect(defined?(ClusterKit::Config)).to eq('constant')
    end
    
    it 'autoloads Silence' do
      expect(defined?(ClusterKit::Silence)).to eq('constant')
    end
    
    # Don't force load other classes that might not be implemented yet
    it 'has Embedder defined or autoloaded' do
      # Embedder might already be loaded or still autoloaded
      expect(defined?(ClusterKit::Embedder) || ClusterKit.autoload?(:Embedder)).to be_truthy
    end
    
    it 'has SVD defined or autoloaded' do
      # SVD might already be loaded or still autoloaded
      expect(defined?(ClusterKit::SVD) || ClusterKit.autoload?(:SVD)).to be_truthy
    end
    
    it 'has Utils defined or autoloaded' do
      # Utils might already be loaded or still autoloaded
      expect(defined?(ClusterKit::Utils) || ClusterKit.autoload?(:Utils)).to be_truthy
    end
    
    it 'has Preprocessing defined or autoloaded' do
      # Preprocessing might already be loaded or still autoloaded
      expect(defined?(ClusterKit::Preprocessing) || ClusterKit.autoload?(:Preprocessing)).to be_truthy
    end
  end
  
  describe 'configuration' do
    it 'provides configuration access' do
      expect(ClusterKit).to respond_to(:configuration)
      expect(ClusterKit).to respond_to(:configuration=)
    end
    
    it 'provides configure method' do
      expect(ClusterKit).to respond_to(:configure)
    end
    
    it 'initializes configuration' do
      expect(ClusterKit.configuration).to be_a(ClusterKit::Configuration)
    end
  end
  
  describe 'module methods' do
    let(:test_data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
    
    describe '.umap' do
      it 'responds to umap method' do
        expect(ClusterKit).to respond_to(:umap)
      end
      
      it 'creates an Embedder with umap method' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 2
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = ClusterKit.umap(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(ClusterKit::Embedder).to have_received(:new).with(method: :umap, n_components: 2)
        expect(mock_embedder).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 3
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        ClusterKit.umap(test_data, n_components: 3)
        
        expect(ClusterKit::Embedder).to have_received(:new).with(method: :umap, n_components: 3)
      end
      
      it 'passes additional options to Embedder' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 2,
          n_neighbors: 15,
          min_dist: 0.1
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([])
        
        ClusterKit.umap(test_data, n_neighbors: 15, min_dist: 0.1)
        
        expect(ClusterKit::Embedder).to have_received(:new).with(
          method: :umap, 
          n_components: 2, 
          n_neighbors: 15, 
          min_dist: 0.1
        )
      end
    end
    
    describe '.tsne' do
      it 'responds to tsne method' do
        expect(ClusterKit).to respond_to(:tsne)
      end
      
      it 'creates an Embedder with tsne method' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 2
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = ClusterKit.tsne(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(ClusterKit::Embedder).to have_received(:new).with(method: :tsne, n_components: 2)
        expect(mock_embedder).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 3
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        ClusterKit.tsne(test_data, n_components: 3)
        
        expect(ClusterKit::Embedder).to have_received(:new).with(method: :tsne, n_components: 3)
      end
      
      it 'passes additional options to Embedder' do
        mock_embedder = instance_double(ClusterKit::Embedder)
        allow(ClusterKit::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 2,
          perplexity: 30.0,
          learning_rate: 200.0
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([])
        
        ClusterKit.tsne(test_data, perplexity: 30.0, learning_rate: 200.0)
        
        expect(ClusterKit::Embedder).to have_received(:new).with(
          method: :tsne, 
          n_components: 2, 
          perplexity: 30.0, 
          learning_rate: 200.0
        )
      end
    end
    
    describe '.estimate_dimension' do
      it 'responds to estimate_dimension method' do
        expect(ClusterKit).to respond_to(:estimate_dimension)
      end
      
      it 'delegates to Utils.estimate_intrinsic_dimension' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension).and_return(2.5)
        
        result = ClusterKit.estimate_dimension(test_data)
        
        expect(result).to eq(2.5)
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 10)
      end
      
      it 'accepts k parameter and maps to k_neighbors' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension).and_return(3.2)
        
        result = ClusterKit.estimate_dimension(test_data, k: 15)
        
        expect(result).to eq(3.2)
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 15)
      end
      
      it 'defaults k to 10' do
        allow(ClusterKit::Utils).to receive(:estimate_intrinsic_dimension).and_return(2.0)
        
        ClusterKit.estimate_dimension(test_data)
        
        expect(ClusterKit::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 10)
      end
    end
    
    describe '.svd' do
      it 'responds to svd method' do
        expect(ClusterKit).to respond_to(:svd)
      end
      
      it 'delegates to SVD.randomized_svd' do
        mock_result = [[[1.0]], [2.0], [[3.0]]]
        allow(ClusterKit::SVD).to receive(:randomized_svd).and_return(mock_result)
        
        result = ClusterKit.svd(test_data, 2)
        
        expect(result).to eq(mock_result)
        expect(ClusterKit::SVD).to have_received(:randomized_svd).with(test_data, 2, n_iter: 2)
      end
      
      it 'accepts n_iter parameter' do
        allow(ClusterKit::SVD).to receive(:randomized_svd).and_return([])
        
        ClusterKit.svd(test_data, 2, n_iter: 5)
        
        expect(ClusterKit::SVD).to have_received(:randomized_svd).with(test_data, 2, n_iter: 5)
      end
      
      it 'defaults n_iter to 2' do
        allow(ClusterKit::SVD).to receive(:randomized_svd).and_return([])
        
        ClusterKit.svd(test_data, 3)
        
        expect(ClusterKit::SVD).to have_received(:randomized_svd).with(test_data, 3, n_iter: 2)
      end
    end
  end
end