# frozen_string_literal: true

# Start SimpleCov before loading any application code (unless disabled)
unless ENV['DISABLE_SIMPLECOV']
  require 'simplecov'
  SimpleCov.start do
    add_filter '/spec/'
    add_filter '/vendor/'
    add_filter '/ext/'  # Exclude Rust extension code

    add_group 'Models', 'lib/annembed'
    add_group 'Utilities', 'lib/annembed/utils'

    # Temporarily disable minimum coverage to diagnose hanging issue
    # minimum_coverage 50

    # Use multiple formatters
    if ENV['CI']
      formatter SimpleCov::Formatter::SimpleFormatter
    else
      formatter SimpleCov::Formatter::HTMLFormatter
    end
  end
end

require "bundler/setup"
require "annembed"

# Only load RSpec configuration if RSpec is available
if defined?(RSpec)
  # require_relative "support/output_suppressor" # Temporarily disabled

  # Suppress verbose output from the Rust extension
  # The annembed library uses env_logger which respects RUST_LOG
  ENV['RUST_LOG'] = 'error' unless ENV['RUST_LOG']

  RSpec.configure do |config|
    # Enable flags like --only-failures and --next-failure
    config.example_status_persistence_file_path = ".rspec_status"

    # Disable RSpec exposing methods globally on `Module` and `main`
    config.disable_monkey_patching!

    config.expect_with :rspec do |c|
      c.syntax = :expect
    end
  end
end