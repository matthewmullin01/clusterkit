#!/usr/bin/env ruby
# frozen_string_literal: true

require 'bundler/setup'
require 'annembed'

# Helper method for the example
class Array
  def mostly_in_range?(start_idx, end_idx)
    count_in_range = self.count { |idx| idx >= start_idx && idx <= end_idx }
    count_in_range > self.length * 0.7
  end
end

# HDBSCAN Example: Document Clustering Pipeline
# ==============================================
# This example demonstrates using HDBSCAN for document clustering,
# which is the primary use case for this implementation.

puts "HDBSCAN Document Clustering Example"
puts "=" * 50

# Simulate document embeddings after UMAP reduction
# In a real application, you would:
# 1. Embed documents using a model (e.g., BERT, Sentence Transformers)
# 2. Reduce dimensions with UMAP to ~20D
# 3. Apply HDBSCAN clustering

# Generate synthetic "document embeddings" in 20D space
# These represent documents after UMAP reduction
def generate_document_embeddings
  embeddings = []
  
  # Topic 1: Technology articles (30 documents)
  30.times do
    embedding = Array.new(20) { rand(-1.0..1.0) }
    # Add bias to make them cluster
    embedding[0] += 5.0
    embedding[1] += 5.0
    embeddings << embedding
  end
  
  # Topic 2: Science articles (25 documents)
  25.times do
    embedding = Array.new(20) { rand(-1.0..1.0) }
    # Add different bias
    embedding[2] += 5.0
    embedding[3] += 5.0
    embeddings << embedding
  end
  
  # Topic 3: Business articles (20 documents)
  20.times do
    embedding = Array.new(20) { rand(-1.0..1.0) }
    # Add another bias
    embedding[4] += 5.0
    embedding[5] += 5.0
    embeddings << embedding
  end
  
  # Noise: Off-topic or mixed-topic documents (15 documents)
  15.times do
    embedding = Array.new(20) { rand(-3.0..7.0) }
    embeddings << embedding
  end
  
  embeddings
end

# Generate sample data
embeddings = generate_document_embeddings
puts "Generated #{embeddings.length} document embeddings (20D)"
puts "  - 30 technology articles"
puts "  - 25 science articles"
puts "  - 20 business articles"
puts "  - 15 mixed/off-topic articles"

# Apply HDBSCAN clustering
puts "\nApplying HDBSCAN clustering..."
hdbscan = AnnEmbed::Clustering::HDBSCAN.new(
  min_samples: 5,        # Minimum neighborhood size for density estimation
  min_cluster_size: 10   # Minimum cluster size (smaller clusters become noise)
)

# Fit the model
hdbscan.fit(embeddings)

# Get results
puts "\nClustering Results:"
puts "-" * 30
puts "Topics found: #{hdbscan.n_clusters}"
puts "Unclustered documents: #{hdbscan.n_noise_points} (#{(hdbscan.noise_ratio * 100).round(1)}%)"

# Analyze each cluster
cluster_indices = hdbscan.cluster_indices
cluster_indices.each do |topic_id, doc_indices|
  puts "\nTopic #{topic_id}:"
  puts "  - #{doc_indices.length} documents"
  
  # In a real application, you would:
  # 1. Extract keywords from documents in this cluster
  # 2. Generate topic descriptions
  # 3. Find representative documents
  
  # Simulate topic identification based on our synthetic data
  if doc_indices.mostly_in_range?(0, 29)
    puts "  - Likely topic: Technology"
  elsif doc_indices.mostly_in_range?(30, 54)
    puts "  - Likely topic: Science"
  elsif doc_indices.mostly_in_range?(55, 74)
    puts "  - Likely topic: Business"
  else
    puts "  - Likely topic: Mixed"
  end
end

# Show noise documents (unclustered)
if hdbscan.n_noise_points > 0
  puts "\nUnclustered documents (noise):"
  puts "  - #{hdbscan.n_noise_points} documents"
  puts "  - These may be outliers, mixed-topic, or unique documents"
  puts "  - Consider manual review or different clustering parameters"
end

# Alternative: Use module-level convenience method
puts "\n" + "=" * 50
puts "Alternative: Module-level method"
puts "-" * 30

result = AnnEmbed::Clustering.hdbscan(
  embeddings,
  min_samples: 3,
  min_cluster_size: 8
)

puts "Topics found: #{result[:n_clusters]}"
puts "Noise ratio: #{(result[:noise_ratio] * 100).round(1)}%"

# Practical tips
puts "\n" + "=" * 50
puts "Tips for Document Clustering:"
puts "-" * 30
puts "1. Use UMAP to reduce embeddings to 20-50 dimensions first"
puts "2. Start with min_cluster_size = 10-20 for most document sets"
puts "3. Adjust min_samples based on local density (usually 5-10)"
puts "4. Expect 20-40% noise for diverse document collections"
puts "5. Noise documents often need special handling or re-clustering"