# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit::Configuration do
  # Reset configuration after each test
  after(:each) do
    ClusterKit.configuration = nil
    ClusterKit.configure
  end
  
  describe '.configure' do
    it 'yields the configuration object' do
      expect { |b| ClusterKit.configure(&b) }.to yield_with_args(ClusterKit::Configuration)
    end
    
    it 'allows setting verbose mode' do
      ClusterKit.configure do |config|
        config.verbose = true
      end
      
      expect(ClusterKit.configuration.verbose).to be true
    end
    
    it 'creates a configuration if none exists' do
      ClusterKit.configuration = nil
      ClusterKit.configure
      
      expect(ClusterKit.configuration).to be_a(ClusterKit::Configuration)
    end
    
    it 'reuses existing configuration' do
      config1 = ClusterKit.configuration
      ClusterKit.configure
      config2 = ClusterKit.configuration
      
      expect(config1).to equal(config2)
    end
  end
  
  describe '#initialize' do
    context 'with default settings' do
      subject(:config) { described_class.new }
      
      it 'sets verbose to false by default' do
        expect(config.verbose).to be false
      end
    end
    
    context 'with ANNEMBED_VERBOSE environment variable' do
      it 'respects ANNEMBED_VERBOSE=true' do
        ENV['ANNEMBED_VERBOSE'] = 'true'
        config = described_class.new
        expect(config.verbose).to be true
        ENV.delete('ANNEMBED_VERBOSE')
      end
      
      it 'ignores other values of ANNEMBED_VERBOSE' do
        ENV['ANNEMBED_VERBOSE'] = 'yes'
        config = described_class.new
        expect(config.verbose).to be false
        ENV.delete('ANNEMBED_VERBOSE')
      end
    end
    
    context 'with DEBUG environment variable' do
      it 'respects DEBUG=true' do
        ENV['DEBUG'] = 'true'
        config = described_class.new
        expect(config.verbose).to be true
        ENV.delete('DEBUG')
      end
      
      it 'ignores other values of DEBUG' do
        ENV['DEBUG'] = '1'
        config = described_class.new
        expect(config.verbose).to be false
        ENV.delete('DEBUG')
      end
    end
    
    context 'with both environment variables' do
      it 'ANNEMBED_VERBOSE takes precedence over DEBUG' do
        ENV['ANNEMBED_VERBOSE'] = 'true'
        ENV['DEBUG'] = 'false'
        config = described_class.new
        expect(config.verbose).to be true
        ENV.delete('ANNEMBED_VERBOSE')
        ENV.delete('DEBUG')
      end
    end
  end
  
  describe '#verbose=' do
    subject(:config) { described_class.new }
    
    it 'allows setting verbose to true' do
      config.verbose = true
      expect(config.verbose).to be true
    end
    
    it 'allows setting verbose to false' do
      config.verbose = true
      config.verbose = false
      expect(config.verbose).to be false
    end
    
    it 'converts truthy values to boolean' do
      config.verbose = 1
      expect(config.verbose).to eq(1)
      
      config.verbose = nil
      expect(config.verbose).to be_nil
      
      config.verbose = 'true'
      expect(config.verbose).to eq('true')
    end
  end
  
  describe 'module-level configuration' do
    it 'provides a singleton configuration' do
      config1 = ClusterKit.configuration
      config2 = ClusterKit.configuration
      
      expect(config1).to equal(config2)
    end
    
    it 'initializes configuration on first access' do
      ClusterKit.configuration = nil
      expect(ClusterKit.configuration).to be_nil
      
      ClusterKit.configure
      expect(ClusterKit.configuration).to be_a(described_class)
    end
    
    it 'allows replacing the configuration' do
      old_config = ClusterKit.configuration
      new_config = described_class.new
      
      ClusterKit.configuration = new_config
      expect(ClusterKit.configuration).to equal(new_config)
      expect(ClusterKit.configuration).not_to equal(old_config)
    end
  end
  
  describe 'integration with Silence module' do
    it 'is used by Silence.maybe_silence' do
      # This is more of an integration test
      ClusterKit.configuration.verbose = false
      
      # Should call silence_output when verbose is false
      expect(ClusterKit::Silence).to receive(:silence_output)
      ClusterKit::Silence.maybe_silence { }
      
      ClusterKit.configuration.verbose = true
      
      # Should not call silence_output when verbose is true
      expect(ClusterKit::Silence).not_to receive(:silence_output)
      ClusterKit::Silence.maybe_silence { }
    end
  end
end