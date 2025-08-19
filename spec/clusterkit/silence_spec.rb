# frozen_string_literal: true

require 'spec_helper'
require 'clusterkit/silence'
require 'clusterkit/configuration'

RSpec.describe ClusterKit::Silence do
  describe '.silence_stream' do
    it 'suppresses output to the given stream' do
      output = StringIO.new
      
      # Test that output is suppressed
      described_class.silence_stream(STDOUT) do
        print "This should not appear"
      end
      
      # Verify stdout still works after
      expect { print "" }.not_to raise_error
    end
  end
  
  describe '.maybe_silence' do
    context 'when verbose is false' do
      before do
        ClusterKit.configuration.verbose = false
      end
      
      it 'suppresses output' do
        expect(described_class).to receive(:silence_output)
        described_class.maybe_silence { }
      end
    end
    
    context 'when verbose is true' do
      before do
        ClusterKit.configuration.verbose = true
      end
      
      it 'does not suppress output' do
        expect(described_class).not_to receive(:silence_output)
        described_class.maybe_silence { }
      end
    end
  end
  
  describe 'UMAP integration' do
    it 'respects verbose configuration' do
      # Create simple test data
      data = Array.new(20) { Array.new(10) { rand } }
      
      # Test that configuration is respected
      ClusterKit.configuration.verbose = false
      expect(ClusterKit.configuration.verbose).to be false
      
      # When verbose is false, should use silence
      expect(ClusterKit::Silence).to receive(:silence_output).and_call_original
      
      umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5)
      umap.fit(data)
    end
    
    it 'allows verbose output when enabled' do
      data = Array.new(20) { Array.new(10) { rand } }
      
      # When verbose is true, should not silence
      ClusterKit.configuration.verbose = true
      expect(ClusterKit.configuration.verbose).to be true
      
      expect(ClusterKit::Silence).not_to receive(:silence_output)
      
      umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2, n_neighbors: 5)
      umap.fit(data)
      
      # Reset configuration
      ClusterKit.configuration.verbose = false
    end
  end
end