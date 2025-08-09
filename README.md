# annembed-ruby

High-performance dimensionality reduction for Ruby, powered by the [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate.

## Features

- **Multiple embedding algorithms**: UMAP, t-SNE, LargeVis, and Diffusion Maps
- **High performance**: Leverages Rust's speed and parallelization
- **Easy to use**: Simple, Ruby-like API
- **Flexible**: Supports various input formats and configuration options

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'annembed-ruby'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install annembed-ruby

### Prerequisites

- Ruby 2.7 or higher
- Rust toolchain (for building from source)
- BLAS library (OpenBLAS or Intel MKL recommended)

## Quick Start

```ruby
require 'annembed'

# Generate some sample data (2D array)
data = Array.new(1000) { Array.new(50) { rand } }

# Perform UMAP embedding
embedding = AnnEmbed.umap(data, n_components: 2, n_neighbors: 15)

# Or use the full API
embedder = AnnEmbed::Embedder.new(
  method: :umap,
  n_components: 2,
  n_neighbors: 15,
  min_dist: 0.1
)

embedding = embedder.fit_transform(data)

# Save the model for later use
embedder.save("model.ann")

# Load and transform new data
embedder = AnnEmbed::Embedder.load("model.ann")
new_embedding = embedder.transform(new_data)
```

## Practical Example: Reducing Database Embeddings from 1024 to 64 Dimensions

This is a common use case when working with large language model embeddings (e.g., from OpenAI, Cohere, etc.) that need to be stored efficiently in a database while maintaining their semantic properties.

### Step 1: Training Phase (One-time process)

```ruby
require 'annembed'
require 'json'

# Load your training embeddings from database
# These should be representative of your data distribution
training_embeddings = fetch_embeddings_from_database(limit: 10000)  # Array of 1024-dim vectors

# Create and configure the UMAP model
embedder = AnnEmbed::Embedder.new(
  method: :umap,
  n_components: 64,      # Reduce to 64 dimensions
  n_neighbors: 30,       # Higher for more global structure preservation
  min_dist: 0.05,        # Lower values = tighter clusters
  metric: :euclidean,    # or :cosine for normalized embeddings
  random_seed: 42        # For reproducibility
)

# Train the model (this may take a few minutes for large datasets)
puts "Training UMAP model on #{training_embeddings.length} embeddings..."
reduced_embeddings = embedder.fit_transform(training_embeddings)

# Save the trained model to disk
MODEL_PATH = "models/umap_1024_to_64.ann"
embedder.save(MODEL_PATH)
puts "Model saved to #{MODEL_PATH}"

# Optionally, update your database with the reduced training embeddings
reduced_embeddings.each_with_index do |embedding, idx|
  update_database_embedding(training_ids[idx], embedding)
end
```

### Step 2: Production Use - Single Embedding

```ruby
require 'annembed'

# Load the pre-trained model once (e.g., at application startup)
EMBEDDER = AnnEmbed::Embedder.load("models/umap_1024_to_64.ann")

# Function to reduce a single embedding
def reduce_embedding(high_dim_embedding)
  # Input: 1024-dimensional array
  # Output: 64-dimensional array
  EMBEDDER.transform([high_dim_embedding]).first
end

# Example: Process a new document
document = "Your text content here..."
high_dim_embedding = generate_embedding(document)  # Returns 1024-dim vector
low_dim_embedding = reduce_embedding(high_dim_embedding)

# Store in database
save_to_database(document_id: 123,
                embedding: low_dim_embedding,
                original_embedding: high_dim_embedding)  # Optionally keep original
```

### Step 3: Production Use - Batch Processing

```ruby
# For better performance when processing multiple embeddings
def reduce_embeddings_batch(high_dim_embeddings)
  # Input: Array of 1024-dimensional arrays
  # Output: Array of 64-dimensional arrays
  EMBEDDER.transform(high_dim_embeddings)
end

# Example: Process a batch of new documents
documents = fetch_new_documents(limit: 100)
high_dim_embeddings = documents.map { |doc| generate_embedding(doc.text) }

# Reduce all at once (much faster than one-by-one)
low_dim_embeddings = reduce_embeddings_batch(high_dim_embeddings)

# Bulk insert to database
documents.zip(low_dim_embeddings).each do |doc, embedding|
  save_to_database(document_id: doc.id, embedding: embedding)
end
```

### Step 4: Periodic Model Updates

As your data distribution changes over time, you may want to retrain the model:

```ruby
# Schedule this monthly/quarterly
def update_dimension_reduction_model
  # Get recent embeddings that represent current data distribution
  recent_embeddings = fetch_embeddings_from_database(
    where: "created_at > ?",
    date: 3.months.ago,
    limit: 20000
  )

  # Train new model
  new_embedder = AnnEmbed::Embedder.new(
    method: :umap,
    n_components: 64,
    n_neighbors: 30,
    min_dist: 0.05
  )

  new_embedder.fit_transform(recent_embeddings)

  # Save with timestamp
  model_path = "models/umap_1024_to_64_#{Date.today}.ann"
  new_embedder.save(model_path)

  # Test new model before deploying
  test_embeddings = fetch_test_embeddings()
  new_results = new_embedder.transform(test_embeddings)

  if validate_results(new_results)
    # Update symlink or config to point to new model
    File.symlink(model_path, "models/umap_1024_to_64_current.ann")
  end
end
```

### Performance Considerations

1. **Memory Usage**: Training on 10,000 1024-dim vectors requires ~80MB RAM
2. **Training Time**: Expect 30-60 seconds for 10,000 vectors on modern hardware
3. **Transform Speed**: ~1ms per embedding on single-core, much faster in batches
4. **Model Size**: Saved model is typically 10-50MB depending on training size

### Best Practices

1. **Training Data Selection**: Use a representative sample of your data
2. **Validation**: Always validate reduced embeddings maintain semantic similarity
3. **Batch Processing**: Transform multiple embeddings at once when possible
4. **Model Versioning**: Keep track of model versions and training dates
5. **Fallback Strategy**: Keep original embeddings initially until confident in reduction

### What's in the Saved Model File?

The `.ann` model file contains:
- The trained UMAP transformation parameters
- Nearest neighbor graph structure from training data
- Configuration settings (n_components, n_neighbors, etc.)
- Random state for reproducibility

The model file enables you to:
- Transform new data points using the learned manifold
- Ensure consistent dimensionality reduction across your pipeline
- Share trained models between different services/servers

## Supported Algorithms

### UMAP (Uniform Manifold Approximation and Projection)
```ruby
AnnEmbed.umap(data,
  n_components: 2,
  n_neighbors: 15,
  min_dist: 0.1,
  spread: 1.0
)
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```ruby
AnnEmbed.tsne(data,
  n_components: 2,
  perplexity: 30.0,
  learning_rate: 200.0
)
```

### LargeVis
```ruby
embedder = AnnEmbed::Embedder.new(method: :largevis)
embedding = embedder.fit_transform(data)
```

### Diffusion Maps
```ruby
embedder = AnnEmbed::Embedder.new(method: :diffusion)
embedding = embedder.fit_transform(data)
```

## Additional Utilities

### Estimate Intrinsic Dimension
```ruby
dimension = AnnEmbed.estimate_dimension(data, k: 10)
puts "Estimated intrinsic dimension: #{dimension}"
```

### Randomized SVD
```ruby
u, s, v = AnnEmbed.svd(matrix, k: 50)
```

## Performance Tips

1. **Use Numo::NArray**: Native support provides best performance
2. **Enable parallelization**: Set `n_threads` option
3. **Adjust HNSW parameters**: For large datasets, tune `ef_construction` and `max_nb_connection`
4. **Use appropriate BLAS**: Link with optimized BLAS for your platform

## Examples

See the `examples/` directory for more detailed examples:
- `mnist_embedding.rb`: Embedding MNIST digits
- `text_embedding.rb`: Embedding text data
- `visualization.rb`: Plotting embeddings

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/cpetersen/annembed-ruby.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

This gem wraps the excellent [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate by Jean-Pierre Both.