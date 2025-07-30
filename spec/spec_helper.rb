# frozen_string_literal: true

require "bundler/setup"
require "annembed"
require_relative "support/output_suppressor"

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
  
  # Include the output suppressor
  config.include OutputSuppressor
end