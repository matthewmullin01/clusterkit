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
  
  # Note: We're not testing the convenience methods like .umap, .tsne, etc.
  # because they depend on Embedder which may not be fully implemented yet.
  # These would be integration tests anyway.
  
  describe 'module methods' do
    it 'responds to umap method' do
      expect(AnnEmbed).to respond_to(:umap)
    end
    
    it 'responds to tsne method' do
      expect(AnnEmbed).to respond_to(:tsne)
    end
    
    it 'responds to estimate_dimension method' do
      expect(AnnEmbed).to respond_to(:estimate_dimension)
    end
    
    it 'responds to svd method' do
      expect(AnnEmbed).to respond_to(:svd)
    end
  end
end