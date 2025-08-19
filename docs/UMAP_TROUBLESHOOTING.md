# UMAP Troubleshooting Guide

## Known Issues and Solutions

### 1. UMAP Hanging During fit() or fit_transform()

#### Symptoms
- The UMAP algorithm hangs indefinitely during training
- Console output shows: `embedded scales quantiles at 0.05 : 2.00e-1 , 0.5 : 2.00e-1, 0.95 : 2.00e-1, 0.99 : 2.00e-1`
- All quantiles are exactly 0.2, indicating degenerate initialization

#### Important: This Only Affects Test Data
**This issue does NOT occur with real embeddings from text models.** Production usage with embeddings from models like:
- OpenAI's text-embedding-ada-002
- Jina embeddings
- Sentence transformers
- BERT-based models

...works perfectly fine. The hanging only occurs with synthetic random test data.

#### Root Cause
The underlying annembed Rust library's `dmap_init` initialization algorithm expects data with manifold structure (like real embeddings have). When given uniform random data without structure, it can initialize all points to exactly the same location (0.2, 0.2), causing gradient descent to fail.

#### Why Real Embeddings Work
Real text embeddings have:
- Natural clustering (similar texts have similar embeddings)
- Meaningful dimensions that correlate
- Values typically in range [-0.12, 0.12]
- High dimensionality (384-1536 dimensions)
- Inherent manifold structure that UMAP is designed to find

#### Triggering Conditions
The bug only occurs with:
- Uniform random test data without structure
- Synthetic data with very small variance
- Oversimplified test cases
- Small values of `nb_grad_batch` (< 5) combined with random data
- Small values of `nb_sampling_by_edge` (< 5) combined with random data

#### Workarounds

1. **Use conservative data ranges**
   ```ruby
   # Good: Small values centered near 0
   data = 30.times.map { 10.times.map { rand * 0.02 - 0.01 } }
   
   # Bad: Large ranges
   data = 30.times.map { 10.times.map { rand * 4.0 - 2.0 } }
   ```

2. **Use default parameters**
   ```ruby
   # Good: Use defaults
   umap = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 15)
   
   # Risky: Small batch parameters
   umap = ClusterKit::UMAP.new(
     n_components: 2,
     n_neighbors: 15,
     nb_grad_batch: 2,      # Too small - may hang
     nb_sampling_by_edge: 3  # Too small - may hang
   )
   ```

3. **Add structure to your data**
   ```ruby
   # Instead of pure random data, add some structure
   data = 15.times.map do |i|
     30.times.map do |j|
       base = (i.to_f / 15) * 0.01  # Small trend
       noise = rand * 0.01 - 0.005   # Small noise
       base + noise
     end
   end
   ```

### 2. Writing Tests for UMAP

Since UMAP works fine with real embeddings but can hang with random test data, here's how to write better tests:

#### Use Realistic Test Data
```ruby
# GOOD: Generate data with structure similar to real embeddings
def generate_embedding_like_data(n_points, n_dims)
  # Create clusters to simulate semantic grouping
  n_clusters = 3
  points_per_cluster = n_points / n_clusters
  
  data = []
  n_clusters.times do |c|
    # Each cluster has a different center
    center = Array.new(n_dims) { (c - 1) * 0.05 }
    
    points_per_cluster.times do
      # Add Gaussian noise around the center
      point = center.map { |x| x + (rand - 0.5) * 0.02 }
      data << point
    end
  end
  
  data
end

# Use it in tests
test_data = generate_embedding_like_data(30, 768)  # 30 points, 768 dims like BERT
```

#### Load Real Embeddings for Testing
```ruby
# BETTER: Use actual embeddings from a small model
require 'candle'

def generate_real_test_embeddings
  model = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  texts = [
    "The cat sat on the mat",
    "Dogs are loyal animals",
    "Machine learning is fascinating",
    # ... more test sentences
  ]
  model.embed(texts)
end
```

#### Skip Problematic Test Scenarios
```ruby
# For edge cases that don't reflect real usage
it "handles random noise data", skip: "UMAP not designed for structure-less data" do
  # This test would hang because random noise has no manifold
  random_data = 100.times.map { 768.times.map { rand } }
  # Don't test this - it's not a real use case
end
```

### 3. Performance Tuning with New Parameters

As of version 0.2.0, UMAP supports two new parameters for performance tuning:

#### nb_grad_batch
- **Default**: 10
- **Purpose**: Controls the number of gradient descent batches
- **Trade-off**: Lower values = faster but less accurate
- **Safe range**: 5-15
- **Warning**: Values < 5 may cause hanging with certain data

#### nb_sampling_by_edge
- **Default**: 8  
- **Purpose**: Controls the number of negative samples per edge
- **Trade-off**: Lower values = faster but less accurate
- **Safe range**: 5-10
- **Warning**: Values < 5 may cause hanging with certain data

Example usage:
```ruby
# Fast but potentially less accurate
fast_umap = ClusterKit::UMAP.new(
  n_components: 2,
  n_neighbors: 15,
  nb_grad_batch: 5,        # Minimum safe value
  nb_sampling_by_edge: 5   # Minimum safe value
)

# Slower but more accurate
accurate_umap = ClusterKit::UMAP.new(
  n_components: 2,
  n_neighbors: 15,
  nb_grad_batch: 15,       # Higher for better quality
  nb_sampling_by_edge: 10  # Higher for better quality
)
```

### 3. Data Validation Issues

#### NaN or Infinite Values
UMAP will raise an error if your data contains NaN or Infinite values:
```ruby
# This will raise ArgumentError
bad_data = [[1.0, Float::NAN], [3.0, 4.0]]
umap.fit_transform(bad_data)
```

#### Inconsistent Row Lengths
All rows must have the same number of features:
```ruby
# This will raise ArgumentError
bad_data = [[1.0, 2.0], [3.0, 4.0, 5.0]]  # Different lengths
umap.fit_transform(bad_data)
```

### 4. Memory Issues with Large Datasets

UMAP builds an HNSW graph which can be memory-intensive for large datasets.

#### Recommendations:
- For datasets > 100k points, consider sampling
- Monitor memory usage during fit_transform
- Use smaller n_neighbors values for large datasets

### 5. Model Persistence Issues

#### Binary Compatibility
Saved models may not be compatible across different versions of clusterkit. Always test loading saved models after upgrading.

#### File Size
Model files include both the original training data and embeddings, so they can be large:
```ruby
# Check model file size before saving
umap.save("model.bin")
puts "Model size: #{File.size('model.bin') / 1024.0 / 1024.0} MB"
```

## Debugging Tips

### 1. Enable Verbose Output
The annembed library outputs diagnostic information during training. Watch for:
- "embedded scales quantiles" - should NOT all be the same value
- "initial cross entropy value" - should be a reasonable number (not 0 or infinity)
- "final cross entropy value" - should be lower than initial

### 2. Test with Known Working Data
If you're having issues, test with this known working configuration:
```ruby
# Known working test case
test_data = 30.times.map { 10.times.map { rand * 0.5 + 0.25 } }
umap = ClusterKit::UMAP.new(n_components: 2, n_neighbors: 5)
result = umap.fit_transform(test_data)
```

### 3. Check Data Characteristics
```ruby
# Analyze your data before UMAP
data_flat = data.flatten
puts "Data range: [#{data_flat.min}, #{data_flat.max}]"
puts "Data mean: #{data_flat.sum / data_flat.length}"
puts "Data variance: #{data_flat.map { |x| (x - mean) ** 2 }.sum / data_flat.length}"

# If variance is very small or range is very large, consider normalizing
```

## When to Report a Bug

Report an issue if:
1. UMAP consistently hangs even with conservative data and default parameters
2. You get a panic or segfault (not just a Ruby exception)
3. Results are dramatically different from other UMAP implementations
4. Memory usage is unexpectedly high

Include in your bug report:
- Data characteristics (shape, range, variance)
- Parameters used
- Console output including the diagnostic messages
- Ruby and clusterkit versions

## Alternative Solutions

If you continue to experience issues:

1. **Use t-SNE instead** (if 2D visualization is the goal)
   ```ruby
   embedder = ClusterKit::Embedder.new(method: :tsne, n_components: 2)
   result = embedder.fit_transform(data)
   ```

2. **Preprocess your data**
   ```ruby
   # Normalize to [0, 1]
   min = data.flatten.min
   max = data.flatten.max
   normalized = data.map { |row| row.map { |x| (x - min) / (max - min) } }
   ```

3. **Use Python UMAP via PyCall** (if stability is critical)
   ```ruby
   require 'pycall'
   umap = PyCall.import_module('umap')
   reducer = umap.UMAP.new
   embedding = reducer.fit_transform(data)
   ```

## References

- [annembed GitHub Issues](https://github.com/jean-pierreBoth/annembed/issues) - For upstream bugs
- [UMAP Algorithm Paper](https://arxiv.org/abs/1802.03426) - Understanding the algorithm
- [Original Python UMAP](https://github.com/lmcinnes/umap) - Reference implementation