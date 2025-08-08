#!/usr/bin/env ruby
# frozen_string_literal: true

require "bundler/setup"
require "annembed"
require "numo/narray"

# This example demonstrates basic usage of annembed-ruby
# Note: This is a template - actual functionality will work once Rust implementation is complete

puts "annembed-ruby Basic Usage Example"
puts "=" * 40

# Generate some sample data
# In real usage, this would be your actual high-dimensional data
def generate_sample_data
  # Create 3 clusters in 50D space
  n_samples = 300
  n_features = 50
  
  data = []
  3.times do |i|
    # Each cluster centered at a different point
    center = Numo::DFloat.zeros(n_features)
    center[i*10...(i+1)*10] = 5.0
    
    # Generate points around center
    cluster = Numo::DFloat.new(n_samples/3, n_features).rand_norm
    cluster += center
    data << cluster
  end
  
  Numo::NArray.vstack(data)
end

# Example 1: Quick UMAP embedding
puts "\n1. Quick UMAP embedding:"
begin
  data = generate_sample_data
  embedding = AnnEmbed.umap(data, n_components: 2, n_neighbors: 15)
  puts "   Input shape: #{data.shape}"
  puts "   Output shape: #{embedding.shape}"
rescue => e
  puts "   Not implemented yet: #{e.message}"
end

# Example 2: Using the Embedder class with configuration
puts "\n2. Configured t-SNE embedding:"
begin
  embedder = AnnEmbed::Embedder.new(
    method: :tsne,
    n_components: 2,
    perplexity: 30,
    learning_rate: 200
  )
  
  embedding = embedder.fit_transform(data)
  puts "   Method: #{embedder.method}"
  puts "   Components: #{embedder.n_components}"
rescue => e
  puts "   Not implemented yet: #{e.message}"
end

# Example 3: Estimate intrinsic dimension
puts "\n3. Intrinsic dimension estimation:"
begin
  dimension = AnnEmbed.estimate_dimension(data, k: 10)
  puts "   Estimated dimension: #{dimension}"
rescue => e
  puts "   Not implemented yet: #{e.message}"
end

# Example 4: Data preprocessing
puts "\n4. Data preprocessing:"
begin
  # Normalize data before embedding
  normalized = AnnEmbed::Preprocessing.normalize(data, method: :standard)
  puts "   Normalized data shape: #{normalized.shape}"
  puts "   Mean: #{normalized.mean.round(4)}"
  puts "   Std: #{normalized.stddev.round(4)}"
rescue => e
  puts "   Error: #{e.message}"
end

# Example 5: Save and load models (once implemented)
puts "\n5. Model persistence:"
begin
  # Fit a model
  embedder = AnnEmbed::Embedder.new(method: :umap)
  embedder.fit(data)
  
  # Save it
  embedder.save("my_model.ann")
  puts "   Model saved to my_model.ann"
  
  # Load it back
  loaded = AnnEmbed::Embedder.load("my_model.ann")
  puts "   Model loaded successfully"
rescue => e
  puts "   Not implemented yet: #{e.message}"
end

puts "\n" + "=" * 40
puts "Note: Most functionality requires Rust implementation"
puts "See NEXT_STEPS.md for implementation plan"