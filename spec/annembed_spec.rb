# frozen_string_literal: true

require 'spec_helper'

RSpec.describe AnnEmbed do
  describe 'module constants' do
    it 'defines METHODS constant' do
      expect(AnnEmbed::METHODS).to be_a(Array)
      expect(AnnEmbed::METHODS).to include(:umap, :tsne, :largevis, :diffusion)
    end
    
    it 'has frozen METHODS array' do
      expect(AnnEmbed::METHODS).to be_frozen
    end
  end
  
  describe 'error classes' do
    it 'defines Error as StandardError subclass' do
      expect(AnnEmbed::Error).to be < StandardError
    end
    
    it 'defines DimensionError' do
      expect(AnnEmbed::DimensionError).to be < AnnEmbed::Error
    end
    
    it 'defines ConvergenceError' do
      expect(AnnEmbed::ConvergenceError).to be < AnnEmbed::Error
    end
    
    it 'defines InvalidParameterError' do
      expect(AnnEmbed::InvalidParameterError).to be < AnnEmbed::Error
    end
    
    it 'allows raising custom errors' do
      expect { raise AnnEmbed::DimensionError, "test" }.to raise_error(AnnEmbed::DimensionError, "test")
    end
  end
  
  describe 'autoloaded classes' do
    it 'autoloads UMAP' do
      expect(defined?(AnnEmbed::UMAP)).to eq('constant')
    end
    
    it 'autoloads Config' do
      expect(defined?(AnnEmbed::Config)).to eq('constant')
    end
    
    it 'autoloads Silence' do
      expect(defined?(AnnEmbed::Silence)).to eq('constant')
    end
    
    # Don't force load other classes that might not be implemented yet
    it 'has Embedder defined or autoloaded' do
      # Embedder might already be loaded or still autoloaded
      expect(defined?(AnnEmbed::Embedder) || AnnEmbed.autoload?(:Embedder)).to be_truthy
    end
    
    it 'has SVD defined or autoloaded' do
      # SVD might already be loaded or still autoloaded
      expect(defined?(AnnEmbed::SVD) || AnnEmbed.autoload?(:SVD)).to be_truthy
    end
    
    it 'has Utils defined or autoloaded' do
      # Utils might already be loaded or still autoloaded
      expect(defined?(AnnEmbed::Utils) || AnnEmbed.autoload?(:Utils)).to be_truthy
    end
    
    it 'has Preprocessing defined or autoloaded' do
      # Preprocessing might already be loaded or still autoloaded
      expect(defined?(AnnEmbed::Preprocessing) || AnnEmbed.autoload?(:Preprocessing)).to be_truthy
    end
  end
  
  describe 'configuration' do
    it 'provides configuration access' do
      expect(AnnEmbed).to respond_to(:configuration)
      expect(AnnEmbed).to respond_to(:configuration=)
    end
    
    it 'provides configure method' do
      expect(AnnEmbed).to respond_to(:configure)
    end
    
    it 'initializes configuration' do
      expect(AnnEmbed.configuration).to be_a(AnnEmbed::Configuration)
    end
  end
  
  describe 'module methods' do
    let(:test_data) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }
    
    describe '.umap' do
      it 'responds to umap method' do
        expect(AnnEmbed).to respond_to(:umap)
      end
      
      it 'creates an Embedder with umap method' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 2
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = AnnEmbed.umap(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(AnnEmbed::Embedder).to have_received(:new).with(method: :umap, n_components: 2)
        expect(mock_embedder).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 3
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        AnnEmbed.umap(test_data, n_components: 3)
        
        expect(AnnEmbed::Embedder).to have_received(:new).with(method: :umap, n_components: 3)
      end
      
      it 'passes additional options to Embedder' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :umap, 
          n_components: 2,
          n_neighbors: 15,
          min_dist: 0.1
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([])
        
        AnnEmbed.umap(test_data, n_neighbors: 15, min_dist: 0.1)
        
        expect(AnnEmbed::Embedder).to have_received(:new).with(
          method: :umap, 
          n_components: 2, 
          n_neighbors: 15, 
          min_dist: 0.1
        )
      end
    end
    
    describe '.tsne' do
      it 'responds to tsne method' do
        expect(AnnEmbed).to respond_to(:tsne)
      end
      
      it 'creates an Embedder with tsne method' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 2
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = AnnEmbed.tsne(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(AnnEmbed::Embedder).to have_received(:new).with(method: :tsne, n_components: 2)
        expect(mock_embedder).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 3
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        AnnEmbed.tsne(test_data, n_components: 3)
        
        expect(AnnEmbed::Embedder).to have_received(:new).with(method: :tsne, n_components: 3)
      end
      
      it 'passes additional options to Embedder' do
        mock_embedder = instance_double(AnnEmbed::Embedder)
        allow(AnnEmbed::Embedder).to receive(:new).with(
          method: :tsne, 
          n_components: 2,
          perplexity: 30.0,
          learning_rate: 200.0
        ).and_return(mock_embedder)
        allow(mock_embedder).to receive(:fit_transform).and_return([])
        
        AnnEmbed.tsne(test_data, perplexity: 30.0, learning_rate: 200.0)
        
        expect(AnnEmbed::Embedder).to have_received(:new).with(
          method: :tsne, 
          n_components: 2, 
          perplexity: 30.0, 
          learning_rate: 200.0
        )
      end
    end
    
    describe '.estimate_dimension' do
      it 'responds to estimate_dimension method' do
        expect(AnnEmbed).to respond_to(:estimate_dimension)
      end
      
      it 'delegates to Utils.estimate_intrinsic_dimension' do
        allow(AnnEmbed::Utils).to receive(:estimate_intrinsic_dimension).and_return(2.5)
        
        result = AnnEmbed.estimate_dimension(test_data)
        
        expect(result).to eq(2.5)
        expect(AnnEmbed::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 10)
      end
      
      it 'accepts k parameter and maps to k_neighbors' do
        allow(AnnEmbed::Utils).to receive(:estimate_intrinsic_dimension).and_return(3.2)
        
        result = AnnEmbed.estimate_dimension(test_data, k: 15)
        
        expect(result).to eq(3.2)
        expect(AnnEmbed::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 15)
      end
      
      it 'defaults k to 10' do
        allow(AnnEmbed::Utils).to receive(:estimate_intrinsic_dimension).and_return(2.0)
        
        AnnEmbed.estimate_dimension(test_data)
        
        expect(AnnEmbed::Utils).to have_received(:estimate_intrinsic_dimension).with(test_data, k_neighbors: 10)
      end
    end
    
    describe '.svd' do
      it 'responds to svd method' do
        expect(AnnEmbed).to respond_to(:svd)
      end
      
      it 'delegates to SVD.randomized_svd' do
        mock_result = [[[1.0]], [2.0], [[3.0]]]
        allow(AnnEmbed::SVD).to receive(:randomized_svd).and_return(mock_result)
        
        result = AnnEmbed.svd(test_data, 2)
        
        expect(result).to eq(mock_result)
        expect(AnnEmbed::SVD).to have_received(:randomized_svd).with(test_data, 2, n_iter: 2)
      end
      
      it 'accepts n_iter parameter' do
        allow(AnnEmbed::SVD).to receive(:randomized_svd).and_return([])
        
        AnnEmbed.svd(test_data, 2, n_iter: 5)
        
        expect(AnnEmbed::SVD).to have_received(:randomized_svd).with(test_data, 2, n_iter: 5)
      end
      
      it 'defaults n_iter to 2' do
        allow(AnnEmbed::SVD).to receive(:randomized_svd).and_return([])
        
        AnnEmbed.svd(test_data, 3)
        
        expect(AnnEmbed::SVD).to have_received(:randomized_svd).with(test_data, 3, n_iter: 2)
      end
    end
  end
end