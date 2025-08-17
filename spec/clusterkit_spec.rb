# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit do
  # No more METHODS constant since we only support UMAP directly
  
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
  
  describe 'autoloaded modules and classes' do
    it 'autoloads Dimensionality module' do
      expect(defined?(ClusterKit::Dimensionality)).to eq('constant')
    end
    
    it 'autoloads Clustering module' do
      expect(defined?(ClusterKit::Clustering)).to eq('constant')
    end
    
    it 'provides UMAP through Dimensionality module' do
      expect(defined?(ClusterKit::Dimensionality::UMAP)).to eq('constant')
    end
    
    it 'provides PCA through Dimensionality module' do
      expect(defined?(ClusterKit::Dimensionality::PCA)).to eq('constant')
    end
    
    it 'provides SVD through Dimensionality module' do
      expect(defined?(ClusterKit::Dimensionality::SVD)).to eq('constant')
    end
    
    it 'provides KMeans through Clustering module' do
      expect(defined?(ClusterKit::Clustering::KMeans)).to eq('constant')
    end
    
    it 'provides HDBSCAN through Clustering module' do
      expect(defined?(ClusterKit::Clustering::HDBSCAN)).to eq('constant')
    end
    
    it 'autoloads Silence' do
      expect(defined?(ClusterKit::Silence)).to eq('constant')
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
      
      it 'creates a UMAP instance' do
        mock_umap = instance_double(ClusterKit::Dimensionality::UMAP)
        allow(ClusterKit::Dimensionality::UMAP).to receive(:new).with(
          n_components: 2
        ).and_return(mock_umap)
        allow(mock_umap).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = ClusterKit.umap(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(ClusterKit::Dimensionality::UMAP).to have_received(:new).with(n_components: 2)
        expect(mock_umap).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_umap = instance_double(ClusterKit::Dimensionality::UMAP)
        allow(ClusterKit::Dimensionality::UMAP).to receive(:new).with(
          n_components: 3
        ).and_return(mock_umap)
        allow(mock_umap).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        ClusterKit.umap(test_data, n_components: 3)
        
        expect(ClusterKit::Dimensionality::UMAP).to have_received(:new).with(n_components: 3)
      end
      
      it 'passes additional options to UMAP' do
        mock_umap = instance_double(ClusterKit::Dimensionality::UMAP)
        allow(ClusterKit::Dimensionality::UMAP).to receive(:new).with(
          n_components: 2,
          n_neighbors: 15,
          min_dist: 0.1
        ).and_return(mock_umap)
        allow(mock_umap).to receive(:fit_transform).and_return([])
        
        ClusterKit.umap(test_data, n_neighbors: 15, min_dist: 0.1)
        
        expect(ClusterKit::Dimensionality::UMAP).to have_received(:new).with(
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
      
      it 'raises NotImplementedError' do
        expect { ClusterKit.tsne(test_data) }.to raise_error(
          NotImplementedError, 
          /t-SNE is not yet implemented/
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
      
      it 'creates an SVD instance and performs fit_transform' do
        mock_result = [[[1.0]], [2.0], [[3.0]]]
        mock_svd = instance_double(ClusterKit::Dimensionality::SVD)
        allow(ClusterKit::Dimensionality::SVD).to receive(:new).with(
          n_components: 2,
          n_iter: 2
        ).and_return(mock_svd)
        allow(mock_svd).to receive(:fit_transform).and_return(mock_result)
        
        result = ClusterKit.svd(test_data, 2)
        
        expect(result).to eq(mock_result)
        expect(ClusterKit::Dimensionality::SVD).to have_received(:new).with(n_components: 2, n_iter: 2)
        expect(mock_svd).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_iter parameter' do
        mock_svd = instance_double(ClusterKit::Dimensionality::SVD)
        allow(ClusterKit::Dimensionality::SVD).to receive(:new).with(
          n_components: 2,
          n_iter: 5
        ).and_return(mock_svd)
        allow(mock_svd).to receive(:fit_transform).and_return([])
        
        ClusterKit.svd(test_data, 2, n_iter: 5)
        
        expect(ClusterKit::Dimensionality::SVD).to have_received(:new).with(n_components: 2, n_iter: 5)
      end
      
      it 'defaults n_iter to 2' do
        mock_svd = instance_double(ClusterKit::Dimensionality::SVD)
        allow(ClusterKit::Dimensionality::SVD).to receive(:new).with(
          n_components: 3,
          n_iter: 2
        ).and_return(mock_svd)
        allow(mock_svd).to receive(:fit_transform).and_return([])
        
        ClusterKit.svd(test_data, 3)
        
        expect(ClusterKit::Dimensionality::SVD).to have_received(:new).with(n_components: 3, n_iter: 2)
      end
    end
    
    describe '.pca' do
      it 'responds to pca method' do
        expect(ClusterKit).to respond_to(:pca)
      end
      
      it 'creates a PCA instance and performs fit_transform' do
        mock_pca = instance_double(ClusterKit::Dimensionality::PCA)
        allow(ClusterKit::Dimensionality::PCA).to receive(:new).with(
          n_components: 2
        ).and_return(mock_pca)
        allow(mock_pca).to receive(:fit_transform).and_return([[0.1, 0.2]])
        
        result = ClusterKit.pca(test_data)
        
        expect(result).to eq([[0.1, 0.2]])
        expect(ClusterKit::Dimensionality::PCA).to have_received(:new).with(n_components: 2)
        expect(mock_pca).to have_received(:fit_transform).with(test_data)
      end
      
      it 'accepts n_components parameter' do
        mock_pca = instance_double(ClusterKit::Dimensionality::PCA)
        allow(ClusterKit::Dimensionality::PCA).to receive(:new).with(
          n_components: 3
        ).and_return(mock_pca)
        allow(mock_pca).to receive(:fit_transform).and_return([[0.1, 0.2, 0.3]])
        
        ClusterKit.pca(test_data, n_components: 3)
        
        expect(ClusterKit::Dimensionality::PCA).to have_received(:new).with(n_components: 3)
      end
    end
    
    describe '.kmeans' do
      it 'responds to kmeans method' do
        expect(ClusterKit).to respond_to(:kmeans)
      end
      
      it 'creates a KMeans instance with specified k' do
        mock_kmeans = instance_double(ClusterKit::Clustering::KMeans)
        allow(ClusterKit::Clustering::KMeans).to receive(:new).with(
          k: 3
        ).and_return(mock_kmeans)
        allow(mock_kmeans).to receive(:fit_predict).and_return([0, 1, 2])
        
        result = ClusterKit.kmeans(test_data, k: 3)
        
        expect(result).to eq([0, 1, 2])
        expect(ClusterKit::Clustering::KMeans).to have_received(:new).with(k: 3)
        expect(mock_kmeans).to have_received(:fit_predict).with(test_data)
      end
      
      it 'auto-detects k when not specified' do
        allow(ClusterKit::Clustering::KMeans).to receive(:optimal_k).with(
          test_data, k_range: 2..10
        ).and_return(2)
        
        mock_kmeans = instance_double(ClusterKit::Clustering::KMeans)
        allow(ClusterKit::Clustering::KMeans).to receive(:new).with(
          k: 2
        ).and_return(mock_kmeans)
        allow(mock_kmeans).to receive(:fit_predict).and_return([0, 0, 1])
        
        result = ClusterKit.kmeans(test_data)
        
        expect(result).to eq([0, 0, 1])
        expect(ClusterKit::Clustering::KMeans).to have_received(:optimal_k)
        expect(ClusterKit::Clustering::KMeans).to have_received(:new).with(k: 2)
      end
    end
  end
end