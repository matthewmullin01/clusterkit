# frozen_string_literal: true

module ClusterKit
  class << self
    attr_accessor :configuration
  end

  def self.configure
    self.configuration ||= Configuration.new
    yield(configuration) if block_given?
  end

  class Configuration
    attr_accessor :verbose

    def initialize
      # Default to quiet unless explicitly set or debug env var is present
      @verbose = ENV['ANNEMBED_VERBOSE'] == 'true' || ENV['DEBUG'] == 'true'
    end
  end
end

# Initialize default configuration
AnnEmbed.configure