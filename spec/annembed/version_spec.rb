# frozen_string_literal: true

require 'spec_helper'

RSpec.describe AnnEmbed do
  describe 'VERSION' do
    it 'has a version number' do
      expect(AnnEmbed::VERSION).not_to be_nil
    end
    
    it 'follows semantic versioning format' do
      expect(AnnEmbed::VERSION).to match(/^\d+\.\d+\.\d+/)
    end
    
    it 'is a frozen string' do
      expect(AnnEmbed::VERSION).to be_frozen
    end
  end
end