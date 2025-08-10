# frozen_string_literal: true

SimpleCov.configure do
  # Add custom groups
  add_group 'Core', 'lib/annembed/embedder'
  add_group 'UMAP', 'lib/annembed/umap'
  add_group 'Utils', 'lib/annembed/utils'
  add_group 'Configuration', 'lib/annembed/config'
  
  # Track branches as well as lines
  enable_coverage :branch
  
  # Set thresholds (lowered during development)
  minimum_coverage line: 50, branch: 40
  
  # Don't refuse to run tests if coverage drops (during development)
  # refuse_coverage_drop
  
  # Maximum coverage drop allowed
  maximum_coverage_drop 5
  
  # Configure output directory
  coverage_dir 'coverage'
  
  # Track test files separately
  track_files 'lib/**/*.rb'
  
  # Custom filters
  add_filter do |source_file|
    # Skip version file
    source_file.filename.include?('version.rb')
  end
  
  # Include timestamp in coverage report
  SimpleCov.formatter = SimpleCov::Formatter::MultiFormatter.new([
    SimpleCov::Formatter::HTMLFormatter,
  ])
  
  # Set project name
  command_name 'RSpec'
  
  # Merge results from multiple test runs
  use_merging true
  
  # Set result cache timeout (in seconds)
  merge_timeout 3600
end