# frozen_string_literal: true

require 'spec_helper'

RSpec.describe ClusterKit do
  describe 'VERSION' do
    it 'has a version number' do
      expect(ClusterKit::VERSION).not_to be_nil
    end
    
    it 'follows semantic versioning format' do
      expect(ClusterKit::VERSION).to match(/^\d+\.\d+\.\d+/)
    end
    
    it 'is a frozen string' do
      expect(ClusterKit::VERSION).to be_frozen
    end
  end
end