#!/usr/bin/env ruby
# Example demonstrating how to control verbose output from clusterkit

require 'bundler/setup'
require 'clusterkit'

# Generate some random test data
data = Array.new(50) { Array.new(20) { rand } }

puts "=" * 60
puts "clusterkit Verbose Output Control Demo"
puts "=" * 60

puts "\n1. Default behavior (quiet mode):"
puts "-" * 40
umap1 = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 10)
result1 = umap1.fit_transform(data)
puts "✓ UMAP completed silently"
puts "  Result shape: #{result1.length} x #{result1.first.length}"

puts "\n2. Enable verbose output:"
puts "-" * 40
ClusterKit.configure do |config|
  config.verbose = true
end

umap2 = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 10)
puts "Running UMAP with verbose output enabled..."
result2 = umap2.fit_transform(data)
puts "✓ UMAP completed with debug output"

puts "\n3. Back to quiet mode:"
puts "-" * 40
ClusterKit.configuration.verbose = false

umap3 = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 10)
result3 = umap3.fit_transform(data)
puts "✓ UMAP completed silently again"

puts "\n" + "=" * 60
puts "You can also set verbose mode via environment variable:"
puts "  ANNEMBED_VERBOSE=true ruby your_script.rb"
puts "=" * 60