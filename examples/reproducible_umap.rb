#!/usr/bin/env ruby
# Example: Achieving reproducibility with UMAP despite random seed issues

require_relative '../lib/clusterkit'
require 'json'

# Due to upstream limitations, UMAP doesn't give perfectly reproducible results
# even with a fixed random_seed. Here are workarounds:

# Generate sample data
srand(42)
data = []
3.times do |cluster|
  center = Array.new(50) { rand * 0.1 + cluster * 2.0 }
  30.times do
    point = center.map { |c| c + (rand - 0.5) * 0.3 }
    data << point
  end
end

puts "Workaround 1: Cache transformed results"
puts "=" * 60

# First run: transform and save results
cache_file = "umap_results_cache.json"
if File.exist?(cache_file)
  puts "Loading cached results from #{cache_file}"
  embedded = JSON.parse(File.read(cache_file))
else
  puts "No cache found, running UMAP..."
  umap = ClusterKit::Dimensionality::UMAP.new(
    n_components: 2,
    n_neighbors: 5,
    random_seed: 42  # Still use for *some* consistency
  )
  embedded = umap.fit_transform(data)
  
  # Save results for reproducibility
  File.write(cache_file, JSON.pretty_generate(embedded))
  puts "Results cached to #{cache_file}"
end

puts "First 3 points:"
embedded[0..2].each_with_index do |point, i|
  puts "  Point #{i}: [#{point[0].round(3)}, #{point[1].round(3)}]"
end

puts "\nWorkaround 2: Save and load fitted models"
puts "=" * 60

model_file = "umap_model.bin"

# Train and save model once
if File.exist?(model_file)
  puts "Loading existing model from #{model_file}"
  umap = ClusterKit::Dimensionality::UMAP.load(model_file)
else
  puts "Training new model..."
  umap = ClusterKit::Dimensionality::UMAP.new(
    n_components: 2,
    n_neighbors: 5,
    random_seed: 42
  )
  umap.fit(data)
  umap.save(model_file)
  puts "Model saved to #{model_file}"
end

# Now transform new data with the same model
new_data = data[0..9]  # Take first 10 points as "new" data
transformed = umap.transform(new_data)
puts "Transformed 10 new points with saved model"
puts "First 3 transformed points:"
transformed[0..2].each_with_index do |point, i|
  puts "  Point #{i}: [#{point[0].round(3)}, #{point[1].round(3)}]"
end

puts "\nWorkaround 3: Use PCA for deterministic reduction"
puts "=" * 60

# PCA is deterministic - same input always gives same output
pca = ClusterKit::Dimensionality::PCA.new(n_components: 2)
pca_result1 = pca.fit_transform(data)
pca_result2 = pca.fit_transform(data)  # Do it again

puts "PCA results are identical: #{pca_result1[0] == pca_result2[0]}"
puts "First point from run 1: [#{pca_result1[0][0].round(3)}, #{pca_result1[0][1].round(3)}]"
puts "First point from run 2: [#{pca_result2[0][0].round(3)}, #{pca_result2[0][1].round(3)}]"

puts "\nRecommendations:"
puts "-" * 40
puts "1. For production pipelines, cache UMAP results"
puts "2. For model deployment, save fitted models and reuse them"
puts "3. For testing/CI, use PCA or cached test data"
puts "4. Accept small variations in UMAP results as normal"

# Clean up example files (uncomment to remove)
# File.delete(cache_file) if File.exist?(cache_file)
# File.delete(model_file) if File.exist?(model_file)