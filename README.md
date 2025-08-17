# ClusterKit

A high-performance clustering and dimensionality reduction toolkit for Ruby, powered by best-in-class Rust implementations.

## 🙏 Acknowledgments & Attribution

ClusterKit builds upon excellent work from the Rust ecosystem:

- **[annembed](https://github.com/jean-pierreBoth/annembed)** - Provides the core UMAP, t-SNE, and other dimensionality reduction algorithms. Created by Jean-Pierre Both.
- **[hdbscan](https://github.com/tom-whitehead/hdbscan)** - Provides the HDBSCAN density-based clustering implementation. A Rust port of the original HDBSCAN algorithm.

This gem would not be possible without these foundational libraries. Please consider starring their repositories if you find ClusterKit useful.

## Features

- **Multiple Dimensionality Reduction Algorithms**:
  - UMAP (Uniform Manifold Approximation and Projection)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - LargeVis
  - Diffusion Maps
  - PCA (Principal Component Analysis)
  - SVD (Singular Value Decomposition)

- **Advanced Clustering**:
  - K-means clustering with automatic k selection via elbow method
  - HDBSCAN (Hierarchical Density-Based Spatial Clustering) for density-based clustering with noise detection
  - Silhouette scoring for cluster quality evaluation

- **High Performance**:
  - Leverages Rust's speed and parallelization
  - Efficient memory usage
  - Support for large datasets

- **Easy to Use**:
  - Simple, scikit-learn-like API
  - Consistent interface across algorithms
  - Comprehensive documentation and examples

- **Visualization Tools**:
  - Interactive HTML visualizations
  - Comparison of different algorithms
  - Built-in rake tasks for quick experimentation

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'clusterkit'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install clusterkit

### Prerequisites

- Ruby 2.7 or higher
- Rust toolchain (for building from source)

## Quick Start - Interactive Example

Copy and paste this into IRB to try out the main features:

```ruby
require 'clusterkit'

# Generate sample high-dimensional data with structure
# This simulates real-world data like text embeddings or image features
puts "Creating sample data: 100 points in 50 dimensions with 3 clusters"

# Create data with some inherent structure (3 clusters)
data = []
3.times do |cluster|
  # Each cluster has a different center
  center = Array.new(50) { rand * 0.3 + cluster * 0.3 }
  
  # Add 33 points around each center with some noise
  33.times do
    point = center.map { |c| c + (rand - 0.5) * 0.2 }
    data << point
  end
end

# Add one more point to make it 100
data << Array.new(50) { rand }

# ============================================================
# 1. DIMENSIONALITY REDUCTION - Visualize high-dim data in 2D
# ============================================================

puts "\n1. DIMENSIONALITY REDUCTION:"

# UMAP - Best for preserving both local and global structure
puts "Running UMAP..."
umap = ClusterKit::Embedder.new(method: :umap, n_components: 2)
umap_result = umap.fit_transform(data)
puts "  ✓ Reduced to #{umap_result.first.size}D: #{umap_result[0..2].map { |p| p.map { |v| v.round(3) } }}"

# PCA - Fast linear reduction, good for finding main variations
puts "Running PCA..."
pca = ClusterKit::PCA.new(n_components: 2)
pca_result = pca.fit_transform(data)
puts "  ✓ Reduced to #{pca_result.first.size}D: #{pca_result[0..2].map { |p| p.map { |v| v.round(3) } }}"
puts "  ✓ Explained variance: #{(pca.explained_variance_ratio.sum * 100).round(1)}%"

# ============================================================
# 2. CLUSTERING - Find groups in your data
# ============================================================

puts "\n2. CLUSTERING:"

# K-means - When you know roughly how many clusters to expect
puts "Running K-means..."
# First, find optimal k using elbow method
elbow_scores = ClusterKit::Clustering.elbow_method(umap_result, k_range: 2..6)
optimal_k = ClusterKit::Clustering.detect_optimal_k(elbow_scores)
puts "  ✓ Optimal k detected: #{optimal_k}"

kmeans = ClusterKit::Clustering::KMeans.new(k: optimal_k)
kmeans_labels = kmeans.fit_predict(umap_result)
puts "  ✓ Found #{kmeans_labels.uniq.size} clusters"

# HDBSCAN - When you don't know the number of clusters and have noise
puts "Running HDBSCAN..."
hdbscan = ClusterKit::Clustering::HDBSCAN.new(min_samples: 5, min_cluster_size: 10)
hdbscan_labels = hdbscan.fit_predict(umap_result)
puts "  ✓ Found #{hdbscan.n_clusters} clusters"
puts "  ✓ Identified #{hdbscan.n_noise_points} noise points (#{(hdbscan.noise_ratio * 100).round(1)}%)"

# ============================================================
# 3. EVALUATION - How good are the clusters?
# ============================================================

puts "\n3. CLUSTER EVALUATION:"
silhouette = ClusterKit::Clustering.silhouette_score(umap_result, kmeans_labels)
puts "  K-means silhouette score: #{silhouette.round(3)} (closer to 1 is better)"

# Filter noise for HDBSCAN evaluation
non_noise = hdbscan_labels.each_with_index.select { |l, _| l != -1 }.map(&:last)
if non_noise.any?
  filtered_data = non_noise.map { |i| umap_result[i] }
  filtered_labels = non_noise.map { |i| hdbscan_labels[i] }
  hdbscan_silhouette = ClusterKit::Clustering.silhouette_score(filtered_data, filtered_labels)
  puts "  HDBSCAN silhouette score: #{hdbscan_silhouette.round(3)} (excluding noise)"
end

puts "\n✅ All done! Try visualizing with: rake clusterkit:visualize"
```

## Detailed Usage

### Dimensionality Reduction

#### UMAP (Uniform Manifold Approximation and Projection)

```ruby
# Create embedder with UMAP
embedder = ClusterKit::Embedder.new(
  method: :umap,
  n_components: 2,      # Target dimensions
  n_neighbors: 15,      # Number of neighbors
  min_dist: 0.1,       # Minimum distance between points
  spread: 1.0,         # Effective scale of embedded points
  random_seed: 42      # For reproducibility
)

# Fit and transform data
embedded = embedder.fit_transform(data)

# Or fit once and transform multiple datasets
# Example: Split your data into training and test sets
all_data = Array.new(200) { Array.new(50) { rand } }  # Your full dataset
training_data = all_data[0...150]   # First 150 samples for training
test_data = all_data[150..-1]       # Last 50 samples for testing

embedder.fit(training_data)
test_embedded = embedder.transform(test_data)

# Note: The library automatically adjusts n_neighbors if it's too large for your dataset
```

#### PCA (Principal Component Analysis)

```ruby
pca = ClusterKit::PCA.new(n_components: 2)
transformed = pca.fit_transform(data)

# Access explained variance
puts "Explained variance ratio: #{pca.explained_variance_ratio}"
puts "Cumulative explained variance: #{pca.cumulative_explained_variance_ratio}"

# Inverse transform to reconstruct original data
reconstructed = pca.inverse_transform(transformed)
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

```ruby
# t-SNE for non-linear dimensionality reduction
# Best for visualization, preserves local structure
embedder = ClusterKit::Embedder.new(
  method: :tsne,
  n_components: 2,      # Usually 2 or 3 for visualization
  perplexity: 30.0,     # Balance between local and global structure (5-50)
  random_seed: 42
)

tsne_result = embedder.fit_transform(data)
# Note: t-SNE doesn't support transform() on new data - use fit_transform only
```

#### SVD (Singular Value Decomposition)

```ruby
# Direct SVD decomposition
# Returns U, S, V matrices where data ≈ U * S * V^T
u, s, vt = ClusterKit.svd(data, k=10, n_iter: 5)

# U: left singular vectors (documents in LSA)
# S: singular values (importance of each component)
# V^T: right singular vectors (terms in LSA)

puts "Shape of U: #{u.size}x#{u.first.size}"
puts "Singular values: #{s[0..4].map { |v| v.round(2) }}"
puts "Shape of V^T: #{vt.size}x#{vt.first.size}"

# For dimensionality reduction, use U * S
reduced = u.map.with_index do |row, i|
  row.map.with_index { |val, j| val * s[j] }
end
```

#### LargeVis

```ruby
# LargeVis for large-scale visualization
# Efficient for very large datasets
embedder = ClusterKit::Embedder.new(
  method: :largevis,
  n_components: 2,
  n_neighbors: 150,     # More neighbors for large datasets
  n_epochs: 5,          # Number of optimization epochs
  random_seed: 42
)

largevis_result = embedder.fit_transform(large_dataset)
```

#### Diffusion Maps

```ruby
# Diffusion Maps for manifold learning
# Good for discovering intrinsic geometry
embedder = ClusterKit::Embedder.new(
  method: :diffusion,
  n_components: 2,
  n_neighbors: 30,
  alpha: 0.5,           # Diffusion parameter
  random_seed: 42
)

diffusion_result = embedder.fit_transform(data)
```

### Clustering

#### K-means with Automatic K Selection

```ruby
# Find optimal number of clusters
elbow_scores = ClusterKit::Clustering.elbow_method(data, k_range: 2..10)
optimal_k = ClusterKit::Clustering.detect_optimal_k(elbow_scores)

# Cluster with optimal k
kmeans = ClusterKit::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
labels = kmeans.fit_predict(data)

# Access cluster centers
centers = kmeans.cluster_centers
```

#### HDBSCAN (Density-Based Clustering)

```ruby
# HDBSCAN automatically determines the number of clusters
# and can identify noise points
hdbscan = ClusterKit::Clustering::HDBSCAN.new(
  min_samples: 5,        # Minimum samples in neighborhood
  min_cluster_size: 10,  # Minimum cluster size
  metric: 'euclidean'    # Distance metric
)

labels = hdbscan.fit_predict(data)

# Noise points are labeled as -1
puts "Clusters found: #{hdbscan.n_clusters}"
puts "Noise points: #{hdbscan.n_noise_points} (#{(hdbscan.noise_ratio * 100).round(1)}%)"

# Access additional HDBSCAN information
probabilities = hdbscan.probabilities      # Cluster membership probabilities
outlier_scores = hdbscan.outlier_scores   # Outlier scores for each point
```

### Visualization

ClusterKit includes a built-in visualization tool:

```bash
# Generate interactive visualization
rake clusterkit:visualize

# With options
rake clusterkit:visualize[output.html,iris,both]  # filename, dataset, clustering method

# Dataset options: clusters, swiss, iris
# Clustering options: kmeans, hdbscan, both
```

This creates an interactive HTML file with:
- Side-by-side comparison of dimensionality reduction methods
- Clustering results visualization
- Performance metrics
- Interactive Plotly.js charts

## Choosing the Right Algorithm

### Dimensionality Reduction

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **UMAP** | General purpose, preserving both local and global structure | Fast, scalable, supports transform() | Requires tuning parameters |
| **t-SNE** | Visualization, revealing clusters | Excellent for finding local patterns | Slow on large data, no transform() |
| **PCA** | Linear relationships, feature extraction | Very fast, interpretable, deterministic | Only captures linear relationships |
| **SVD** | Text analysis (LSA), recommendation systems | Memory efficient, good for sparse data | Only linear relationships |
| **LargeVis** | Very large datasets (>100k points) | Scales well, faster than t-SNE | Less refined than UMAP |
| **Diffusion Maps** | Non-linear manifold data | Robust to noise, reveals geometry | Computationally intensive |

### Clustering

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **K-means** | Spherical clusters, known cluster count | Fast, simple, deterministic with seed | Requires knowing k, assumes spherical clusters |
| **HDBSCAN** | Unknown cluster count, irregular shapes, noise | Finds clusters automatically, handles noise | More complex parameters, slower than k-means |

### Recommended Combinations

- **Document Clustering**: UMAP (20D) → HDBSCAN
- **Image Clustering**: PCA (50D) → K-means
- **Customer Segmentation**: UMAP (10D) → K-means with elbow method
- **Anomaly Detection**: UMAP (5D) → HDBSCAN (outliers are noise points)
- **Visualization**: UMAP or t-SNE (2D) → visual inspection

## Advanced Examples

### Document Clustering Pipeline

```ruby
# Typical NLP workflow: embed → reduce → cluster
documents = ["text1", "text2", ...]  # Your documents

# Step 1: Get embeddings (use your favorite embedding model)
# embeddings = get_embeddings(documents)  # e.g., from red-candle

# Step 2: Reduce dimensions for better clustering
umap = ClusterKit::UMAP.new(n_components: 20, n_neighbors: 15)
reduced_embeddings = umap.fit_transform(embeddings)

# Step 3: Find clusters
hdbscan = ClusterKit::Clustering::HDBSCAN.new(
  min_samples: 5,
  min_cluster_size: 10
)
clusters = hdbscan.fit_predict(reduced_embeddings)

# Step 4: Analyze results
clusters.each_with_index do |cluster_id, doc_idx|
  next if cluster_id == -1  # Skip noise
  puts "Document '#{documents[doc_idx]}' belongs to cluster #{cluster_id}"
end
```

### Model Persistence

```ruby
# Save trained model
embedder.save("model.bin")

# Load trained model
loaded_embedder = ClusterKit::Embedder.load("model.bin")
result = loaded_embedder.transform(new_data)
```

## Performance Tips

1. **Large Datasets**: Use sampling for initial parameter tuning
2. **HDBSCAN**: Reduce to 10-50 dimensions with UMAP first for better results
3. **Memory**: Process in batches for very large datasets
4. **Speed**: Compile with optimizations: `RUSTFLAGS="-C target-cpu=native" bundle install`

## Troubleshooting

### UMAP "isolated point" or "graph not connected" errors

This error occurs when UMAP cannot find enough neighbors for some points. Solutions:

1. **Reduce n_neighbors**: Use a smaller value (e.g., 5 instead of 15)
   ```ruby
   embedder = ClusterKit::Embedder.new(method: :umap, n_neighbors: 5)
   ```

2. **Add structure to your data**: Completely random data may not work well
   ```ruby
   # Instead of: data = Array.new(100) { Array.new(50) { rand } }
   # Use data with some structure (see Quick Start example)
   ```

3. **Ensure sufficient data points**: UMAP needs at least n_neighbors + 1 points

### Memory issues with large datasets

- Process in batches for datasets > 100k points
- Use PCA to reduce dimensions before UMAP
- Consider using LargeVis instead of UMAP for very large datasets

### Installation issues

- Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- For M1/M2 Macs, ensure you have the latest Xcode command line tools
- Clear the build cache if needed: `bundle exec rake clean`

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests.

To install this gem onto your local machine, run `bundle exec rake install`.

## Testing

```bash
# Run all tests
bundle exec rspec

# Run specific test file
bundle exec rspec spec/clusterkit/clustering_spec.rb

# Run with coverage
COVERAGE=true bundle exec rspec
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/cpetersen/clusterkit.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Citation

If you use ClusterKit in your research, please cite:

```
@software{clusterkit,
  author = {Chris Petersen},
  title = {ClusterKit: High-Performance Clustering and Dimensionality Reduction for Ruby},
  year = {2024},
  url = {https://github.com/cpetersen/clusterkit}
}
```

And please also cite the underlying libraries:
- [annembed](https://github.com/jean-pierreBoth/annembed) for dimensionality reduction algorithms
- [hdbscan](https://github.com/petabi/hdbscan) for HDBSCAN clustering