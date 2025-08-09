# annembed-ruby

High-performance dimensionality reduction for Ruby, powered by the [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate.

## Features

- **UMAP algorithm**: State-of-the-art dimensionality reduction
- **High performance**: Leverages Rust's speed and parallelization
- **Easy to use**: Simple, scikit-learn-like API
- **Model persistence**: Save and load trained models
- **Flexible**: Supports various configuration options

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

## Quick Start

```ruby
require 'annembed'

# Generate some sample data (2D array)
data = Array.new(100) { Array.new(50) { rand } }

# Create a UMAP instance
umap = AnnEmbed::UMAP.new(n_components: 2, n_neighbors: 15)

# Fit and transform in one step
embedding = umap.fit_transform(data)

# Or fit and transform separately
umap.fit(data)
embedding = umap.transform(data)

# Check if model is fitted
puts "Model fitted: #{umap.fitted?}"

# Save the model for later use
umap.save("model.bin")

# Load and use a saved model
loaded_umap = AnnEmbed::UMAP.load("model.bin")
new_embedding = loaded_umap.transform(new_data)
```

## API Reference

### AnnEmbed::UMAP

The main class for UMAP dimensionality reduction.

#### Initialization

```ruby
umap = AnnEmbed::UMAP.new(
  n_components: 2,    # Target number of dimensions (default: 2)
  n_neighbors: 15,    # Number of neighbors for manifold approximation (default: 15)
  random_seed: 42     # Random seed for reproducibility (optional)
)
```

#### Methods

##### `#fit(data)`
Train the model on the provided data.

```ruby
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
umap.fit(data)  # Returns self for method chaining
```

##### `#transform(data)`
Transform new data using the fitted model.

```ruby
new_data = [[7.0, 8.0], [9.0, 10.0]]
reduced = umap.transform(new_data)  # Returns 2D array of reduced dimensions
```

##### `#fit_transform(data)`
Fit the model and transform the data in one step.

```ruby
embedding = umap.fit_transform(data)  # Returns transformed data
```

##### `#fitted?`
Check if the model has been fitted.

```ruby
if umap.fitted?
  puts "Model is ready to transform data"
end
```

##### `#save(path)`
Save the fitted model to a file.

```ruby
umap.save("my_model.bin")
```

##### `.load(path)`
Load a previously saved model.

```ruby
umap = AnnEmbed::UMAP.load("my_model.bin")
```

#### Data Export/Import Utilities

For caching transformed results:

```ruby
# Export transformed data to JSON
AnnEmbed::UMAP.export_data(embedding, "embeddings.json")

# Import previously exported data
cached_embedding = AnnEmbed::UMAP.import_data("embeddings.json")
```

## Practical Example: Reducing Text Embeddings

A common use case for UMAP is reducing high-dimensional embeddings (e.g., from OpenAI, Cohere, etc.) to lower dimensions for storage and visualization.

### Step 1: Training the Model

```ruby
require 'annembed'
require 'json'

# Load your training embeddings (e.g., 768-dim vectors from a language model)
training_embeddings = load_embeddings_from_database(limit: 10000)

# Create and configure UMAP
umap = AnnEmbed::UMAP.new(
  n_components: 50,     # Reduce to 50 dimensions
  n_neighbors: 30,      # Higher for more global structure
  random_seed: 42       # For reproducibility
)

# Train the model
puts "Training UMAP on #{training_embeddings.length} embeddings..."
reduced_embeddings = umap.fit_transform(training_embeddings)

# Save the trained model
umap.save("models/text_umap_768_to_50.bin")
puts "Model saved!"

# Optionally cache the reduced embeddings
AnnEmbed::UMAP.export_data(reduced_embeddings, "cache/reduced_embeddings.json")
```

### Step 2: Production Use - Single Embedding

```ruby
require 'annembed'

# Load the pre-trained model once (e.g., at application startup)
UMAP_MODEL = AnnEmbed::UMAP.load("models/text_umap_768_to_50.bin")

# Function to reduce a single embedding
def reduce_embedding(high_dim_embedding)
  # Input: 768-dimensional array
  # Output: 50-dimensional array
  UMAP_MODEL.transform([high_dim_embedding]).first
end

# Example usage
document = "Your text content here..."
high_dim_embedding = generate_embedding(document)  # Returns 768-dim vector
low_dim_embedding = reduce_embedding(high_dim_embedding)

# Store in database
save_to_database(
  document_id: 123,
  embedding: low_dim_embedding,
  original_embedding: high_dim_embedding  # Optionally keep original
)
```

### Step 3: Batch Processing

```ruby
# For better performance when processing multiple embeddings
def reduce_embeddings_batch(high_dim_embeddings)
  UMAP_MODEL.transform(high_dim_embeddings)
end

# Example: Process a batch of documents
documents = fetch_new_documents(limit: 100)
high_dim_embeddings = documents.map { |doc| generate_embedding(doc.text) }

# Reduce all at once (much faster than one-by-one)
low_dim_embeddings = reduce_embeddings_batch(high_dim_embeddings)

# Bulk insert to database
documents.zip(low_dim_embeddings).each do |doc, embedding|
  save_to_database(document_id: doc.id, embedding: embedding)
end
```

### Step 4: Caching and Recovery

```ruby
# Cache transformed results for recovery
embeddings = umap.fit_transform(training_data)
AnnEmbed::UMAP.export_data(embeddings, "backup/embeddings_#{Date.today}.json")

# Later, if needed
cached_embeddings = AnnEmbed::UMAP.import_data("backup/embeddings_2024-01-15.json")
```

## Real-World Example: Search System with Reduced Dimensions

```ruby
class EmbeddingService
  def initialize(model_path)
    @umap = AnnEmbed::UMAP.load(model_path)
    @embedder = TextEmbedder.new  # Your text embedding service
  end
  
  def process_document(text)
    # Generate high-dimensional embedding
    full_embedding = @embedder.embed(text)
    
    # Reduce dimensions for efficient storage
    reduced_embedding = @umap.transform([full_embedding]).first
    
    {
      full: full_embedding,      # 768 dimensions
      reduced: reduced_embedding  # 50 dimensions
    }
  end
  
  def search(query, documents)
    # Get query embedding
    query_full = @embedder.embed(query)
    query_reduced = @umap.transform([query_full]).first
    
    # Fast approximate search using reduced dimensions
    candidates = documents.sort_by do |doc|
      cosine_distance(query_reduced, doc[:reduced_embedding])
    end.first(20)
    
    # Rerank using full embeddings for precision
    candidates.sort_by do |doc|
      cosine_distance(query_full, doc[:full_embedding])
    end.first(5)
  end
end
```

## Performance Considerations

1. **Training Data Size**: UMAP works best with at least 100 samples
2. **Memory Usage**: Training on 10,000 768-dim vectors requires ~60MB RAM
3. **Training Time**: Expect 30-60 seconds for 10,000 vectors
4. **Transform Speed**: ~1ms per embedding, faster in batches
5. **Model Size**: Saved models are typically 5-20MB

## Best Practices

1. **Representative Training Data**: Use a diverse sample that represents your data distribution
2. **Validation**: Always validate that reduced embeddings maintain semantic relationships
3. **Batch Processing**: Transform multiple embeddings at once for better performance
4. **Model Versioning**: Keep track of model versions and training dates
5. **Error Handling**: Always check `fitted?` before calling `transform`

```ruby
# Good practice
if umap.fitted?
  result = umap.transform(data)
else
  raise "Model not fitted!"
end

# Batch processing
results = umap.transform(batch_of_100_items)  # Fast

# vs individual processing
results = items.map { |item| umap.transform([item]).first }  # Slower
```

## Working with Different Data Formats

```ruby
# Regular Ruby arrays
data = [[1.0, 2.0], [3.0, 4.0]]
embedding = umap.fit_transform(data)

# From CSV
require 'csv'
data = CSV.read('data.csv').map { |row| row.map(&:to_f) }
embedding = umap.fit_transform(data)

# From JSON
require 'json'
data = JSON.parse(File.read('data.json'))
embedding = umap.fit_transform(data)

# Export results
AnnEmbed::UMAP.export_data(embedding, 'results.json')
```

## Error Handling

The UMAP implementation provides clear error messages:

```ruby
# Empty data
umap.fit([])  # => ArgumentError: Input cannot be empty

# Inconsistent dimensions
umap.fit([[1, 2], [3, 4, 5]])  # => ArgumentError: All rows must have the same length

# Non-numeric data
umap.fit([["a", "b"]])  # => ArgumentError: Element at position [0, 0] is not numeric

# Unfitted model
umap = AnnEmbed::UMAP.new
umap.transform([[1, 2]])  # => RuntimeError: Model must be fitted before transform
```

## Troubleshooting

### "Model must be fitted before transform"
Make sure to call `fit` or `fit_transform` before calling `transform`:

```ruby
umap = AnnEmbed::UMAP.new
umap.fit(training_data)  # Required!
result = umap.transform(new_data)
```

### "assertion failed: (*f).abs() <= box_size"
This internal assertion can occur with data that has extreme values. Normalize your data:

```ruby
# Normalize to [0, 1] range
def normalize(data)
  data.map do |row|
    min = row.min
    max = row.max
    range = max - min
    row.map { |v| range.zero? ? 0.5 : (v - min) / range }
  end
end

normalized_data = normalize(your_data)
embedding = umap.fit_transform(normalized_data)
```

### Poor quality embeddings
- Increase `n_neighbors` for more global structure
- Decrease `n_neighbors` for more local structure
- Ensure you have enough training data (at least 100 samples)
- Check that your input data is meaningful (not random noise)

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yourusername/annembed-ruby.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

This gem wraps the excellent [annembed](https://github.com/jean-pierreBoth/annembed) Rust crate by Jean-Pierre Both.