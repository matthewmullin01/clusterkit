# annembed

High-performance dimensionality reduction for Ruby, powered by the [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate.

## Features

- **Multiple embedding algorithms**: UMAP, t-SNE, LargeVis, and Diffusion Maps
- **High performance**: Leverages Rust's speed and parallelization
- **Easy to use**: Simple, Ruby-like API
- **Flexible**: Supports various input formats and configuration options

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'annembed'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install annembed

### Prerequisites

- Ruby 2.7 or higher
- Rust toolchain (for building from source)
- BLAS library (OpenBLAS or Intel MKL recommended)

## Quick Start

```ruby
require 'annembed'
require 'numo/narray'

# Generate some sample data
data = Numo::DFloat.new(1000, 50).rand_norm

# Perform UMAP embedding
embedding = Annembed.umap(data, n_components: 2, n_neighbors: 15)

# Or use the full API
embedder = Annembed::Embedder.new(
  method: :umap,
  n_components: 2,
  n_neighbors: 15,
  min_dist: 0.1
)

embedding = embedder.fit_transform(data)

# Save the model for later use
embedder.save("model.ann")

# Load and transform new data
embedder = Annembed::Embedder.load("model.ann")
new_embedding = embedder.transform(new_data)
```

## Supported Algorithms

### UMAP (Uniform Manifold Approximation and Projection)
```ruby
Annembed.umap(data, 
  n_components: 2,
  n_neighbors: 15,
  min_dist: 0.1,
  spread: 1.0
)
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```ruby
Annembed.tsne(data,
  n_components: 2,
  perplexity: 30.0,
  learning_rate: 200.0
)
```

### LargeVis
```ruby
embedder = Annembed::Embedder.new(method: :largevis)
embedding = embedder.fit_transform(data)
```

### Diffusion Maps
```ruby
embedder = Annembed::Embedder.new(method: :diffusion)
embedding = embedder.fit_transform(data)
```

## Additional Utilities

### Estimate Intrinsic Dimension
```ruby
dimension = Annembed.estimate_dimension(data, k: 10)
puts "Estimated intrinsic dimension: #{dimension}"
```

### Randomized SVD
```ruby
u, s, v = Annembed.svd(matrix, k: 50)
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

Bug reports and pull requests are welcome on GitHub at https://github.com/yourusername/annembed-ruby.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

This gem wraps the excellent [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate by Jean-Pierre Both.