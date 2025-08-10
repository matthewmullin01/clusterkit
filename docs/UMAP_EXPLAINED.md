# UMAP: Dimensionality Reduction for Software Developers

## What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm that transforms high-dimensional data (e.g., 768-dimensional embeddings) into low-dimensional representations (typically 2D or 3D) while preserving the data's underlying structure. It's particularly effective at maintaining both local neighborhoods and global structure.

## Example: The Sphere Analogy

Consider points in 3D space that form a sphere. You could represent each point with (x, y, z) coordinates, but if you know they lie on a sphere's surface, you could more efficiently represent them with just latitude and longitude - reducing from 3 to 2 dimensions without losing information.

UMAP works similarly: it discovers that your high-dimensional data lies on a lower-dimensional manifold (like the sphere's surface), finds the parameters of that manifold, and maps points to this more efficient coordinate system.

**The key insight**: Just as latitude/longitude preserves relationships between points on Earth's surface, UMAP preserves relationships between points on the discovered manifold. The difference is that UMAP must first discover what shape the manifold is - it might be sphere-like, pretzel-shaped, or something more complex.

```ruby
# Example: 10,000 points on a "sphere" in 100D space
# The data is 100-dimensional, but actually lies on a 2D surface

# Generate data on a hypersphere with noise
points_100d = generate_sphere_surface_in_100d(n_points: 10000)

# UMAP discovers the 2D manifold structure
reducer = AnnEmbed::UMAP.new(n_components: 2)
coords_2d = reducer.fit_transform(points_100d)

# coords_2d now contains the "latitude/longitude" equivalent
# for the discovered manifold - a 2D representation that
# preserves the essential structure of the 100D data
```

This is why UMAP is so effective with embeddings: even though embeddings are high-dimensional, they often lie on or near a much lower-dimensional manifold that captures the true relationships in the data.

## Core Algorithm Concept

UMAP operates on the principle that high-dimensional data lies on a lower-dimensional manifold. It constructs a topological representation of the data and then optimizes a low-dimensional layout to match this topology as closely as possible.

The algorithm makes two key assumptions:
1. The data is uniformly distributed on a Riemannian manifold
2. The manifold is locally connected

## How UMAP Works: Algorithmic Steps

### Step 1: Construct k-NN Graph
- For each data point, find its k nearest neighbors (typically k=15-50)
- Build a weighted graph where edge weights represent distances
- Uses approximate nearest neighbor algorithms for efficiency (e.g., RP-trees, NNDescent)

### Step 2: Compute Fuzzy Simplicial Set
- Convert the k-NN graph into a fuzzy topological representation
- Apply local scaling based on distance to nearest neighbors
- Create symmetric graph by combining directed edges using fuzzy set union

### Step 3: Initialize Low-Dimensional Embedding
- Generate initial positions in target dimension (usually 2D/3D)
- Can use spectral embedding for better initialization or random placement

### Step 4: Optimize Layout via SGD
- Minimize cross-entropy between high-dimensional and low-dimensional fuzzy representations
- Uses attractive forces for connected points, repulsive forces for non-connected points
- Typically runs for 200-500 iterations with learning rate decay

## Technical Classification

**UMAP is:**
- A **non-linear dimensionality reduction technique**
- A **manifold learning algorithm**
- A **graph-based embedding method**

**Algorithm Type:** UMAP doesn't fit traditional ML model categories. It's a transformation algorithm that learns a mapping from high-dimensional to low-dimensional space. Once trained on a dataset, it can transform new points using the learned embedding space.

## Data Requirements

### Dataset Size
- **Minimum viable**: 500 data points
- **Recommended**: 2,000-10,000 points
- **Scales well to**: Millions of points

### Computational Complexity
- **Time complexity**: O(N^1.14) for N data points (approximate)
- **Memory complexity**: O(N) with optimizations
- **Typical runtime**: 30 seconds for 10K points with 100 dimensions

### Input Format
- Dense numerical arrays (numpy arrays, tensors)
- All points must have same dimensionality
- Works best with normalized/standardized features

## When to Use UMAP

### Optimal Use Cases

1. **Embedding Visualization**
   ```
   Input: 10,000 document embeddings (768 dimensions)
   Output: 2D coordinates for plotting
   Purpose: Visualize document clusters and relationships
   ```

2. **Clustering Preprocessing**
   ```
   Input: High-dimensional feature vectors
   Output: 10-50 dimensional representations
   Purpose: Improve clustering algorithm performance and speed
   ```

3. **Anomaly Detection**
   ```
   Input: Normal behavior embeddings
   Output: 2D projection showing outliers
   Purpose: Identify points that don't fit the manifold structure
   ```

4. **Feature Engineering**
   ```
   Input: Raw high-dimensional features
   Output: Lower-dimensional features for downstream ML
   Purpose: Capture non-linear relationships in fewer dimensions
   ```

## Limitations and Alternatives

### UMAP Limitations

1. **Non-deterministic**: Results vary between runs due to:
   - Random initialization
   - Stochastic gradient descent
   - Approximate nearest neighbor search

2. **Distance Distortion**: UMAP preserves topology, not distances
   - Distances in UMAP space don't correspond to original distances
   - Density can be misleading (denser areas might just be artifacts)

3. **Parameter Sensitivity**: Results heavily depend on:
   - `n_neighbors`: Controls local vs global structure balance
   - `min_dist`: Controls cluster tightness
   - `metric`: Distance function choice crucial for certain data types

4. **No Inverse Transform**: Generally cannot reconstruct original data from UMAP coordinates

### When to Use Alternatives

| Scenario | Use Instead | Reason |
|----------|-------------|---------|
| Need exact variance preservation | PCA | PCA preserves maximum variance in linear projections |
| Need deterministic results | PCA, Kernel PCA | These provide reproducible transformations |
| Small dataset (<500 points) | PCA, MDS | UMAP needs sufficient data to learn manifold |
| Need inverse transformation | Autoencoders | Can reconstruct original from embedding |
| Purely categorical data | MCA, FAMD | Designed for categorical/mixed data types |
| Need interpretable dimensions | Factor Analysis, PCA | Dimensions have meaningful interpretations |
| Time series data | DTW + MDS | Respects temporal dependencies |

## Data That Degrades UMAP Performance

### 1. Extreme Sparsity
```ruby
# Problem: 99.9% zeros in data
sparse_data = Array.new(1000) { Array.new(1000) { rand < 0.001 ? 1 : 0 } }
# Solution: Use PCA/SVD first or specialized sparse methods
```

### 2. Curse of Dimensionality
```ruby
# Problem: Dimensions >> samples
data = Array.new(100) { Array.new(10000) { rand } }  # 100 samples, 10000 dimensions
# Solution: Apply PCA first to reduce to ~50 dimensions
```

### 3. Multiple Disconnected Manifolds
```ruby
# Problem: Completely separate clusters with no connections
cluster1 = Array.new(500) { Array.new(100) { rand(-3.0..3.0) } }
cluster2 = Array.new(500) { Array.new(100) { rand(-3.0..3.0) + 1000 } }  # Far separated
# Result: UMAP may arbitrarily position disconnected components
```

### 4. Pure Noise
```ruby
# Problem: No underlying structure
random_data = Array.new(1000) { Array.new(100) { rand(-3.0..3.0) } }
# Result: Meaningless projection, artificial patterns
```

## Key Parameters

### Essential Parameters
```ruby
umap_model = AnnEmbed::UMAP.new(
  n_components: 2,      # Output dimensions
  n_neighbors: 15,      # Number of neighbors (15-50 typical)
  random_seed: 42       # For reproducibility
)
# Note: min_dist and metric parameters may be configurable in future versions
```

### Parameter Effects
- **n_neighbors**:
  - Low (5-15): Preserves local structure, detailed clusters
  - High (50-200): Preserves global structure, broader patterns

- **min_dist**:
  - Near 0: Tight clumps, allows overlapping
  - Near 1: Even distribution, preserves more global structure

- **metric**: Critical for specific data types
  - `euclidean`: Standard for continuous data
  - `cosine`: Text embeddings, directional data
  - `manhattan`: Robust to outliers
  - `hamming`: Binary/categorical features

## Implementation Considerations

### Performance Optimization
```ruby
# For large datasets (>50K points)
umap_model = AnnEmbed::UMAP.new(
  n_neighbors: 15,
  n_components: 2,
  random_seed: 42
)
# The Rust backend automatically optimizes for performance
# using efficient algorithms like HNSW for nearest neighbor search
```

### Supervised vs Unsupervised
```ruby
# Unsupervised: Find natural structure
embedding = umap_model.fit_transform(data)

# Note: Supervised UMAP (using labels to guide projection)
# may be available in future versions of annembed-ruby
```

## Practical Example: Document Embedding Pipeline

```ruby
require 'annembed'
require 'candle'

# Typical workflow for document embeddings
documents = load_documents  # 10,000 documents

# Initialize embedding model using red-candle's from_pretrained
embedding_model = Candle::EmbeddingModel.from_pretrained(
  "jinaai/jina-embeddings-v2-base-en",
  device: Candle::Device.best  # Automatically use GPU if available
)

# Generate embeddings for all documents
# The embedding method returns normalized embeddings by default with "pooled_normalized"
embeddings = documents.map do |doc|
  embedding_model.embedding(doc).to_a  # Convert tensor to array
end
# Shape: Array of 10000 arrays, each with 768 floats

# Embeddings are already normalized when using pooled_normalized (default)
# But if you used a different pooling method, normalize like this:
# embeddings_normalized = embeddings.map do |embedding|
#   magnitude = Math.sqrt(embedding.sum { |x| x**2 })
#   embedding.map { |x| x / magnitude }
# end

# Dimensionality reduction
reducer = AnnEmbed::UMAP.new(
  n_neighbors: 30,
  n_components: 2,
  random_seed: 42
)

# Fit and transform
coords_2d = reducer.fit_transform(embeddings)

# Can now transform new documents
new_doc = "New document"
new_embedding = embedding_model.embedding(new_doc).to_a
new_coords = reducer.transform([new_embedding])

# Alternative: Using different pooling methods
# cls_embedding = embedding_model.embedding(doc, pooling_method: "cls")
# pooled_embedding = embedding_model.embedding(doc, pooling_method: "pooled")
```

## Common Pitfalls

1. **Using raw distances between UMAP points for similarity**
   - Wrong: `distance = Math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)`
   - Right: Use original high-dimensional distances

2. **Not scaling features before UMAP**
   - Features with larger scales dominate distance calculations
   - Always normalize or standardize first

3. **Over-interpreting visual clusters**
   - Clusters in UMAP don't always mean distinct groups
   - Validate with clustering algorithms on original data

4. **Forgetting to normalize embeddings for cosine similarity**
   - Text embeddings typically use cosine distance
   - Normalize vectors before UMAP when working with embeddings

5. **Applying to insufficient data**
   - UMAP needs enough points to learn manifold structure
   - Consider simpler methods for small datasets (< 500 points)

## Complete Ruby Example with Red-Candle

```ruby
require 'candle'
require 'annembed'

# Sample documents for clustering
documents = [
  # Technology cluster
  "Machine learning algorithms process data efficiently",
  "Neural networks enable deep learning applications",
  "Artificial intelligence transforms modern software",

  # Food cluster
  "Italian pasta dishes are delicious and varied",
  "Fresh vegetables make healthy salad options",
  "Chocolate desserts satisfy sweet cravings",

  # Sports cluster
  "Basketball players need excellent coordination",
  "Marathon runners train for endurance events",
  "Tennis matches require mental focus"
]

# Initialize embedding model using from_pretrained
# Model type is auto-detected from the model_id
model = Candle::EmbeddingModel.from_pretrained(
  "sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings (already normalized with pooled_normalized)
embeddings = documents.map { |doc| model.embedding(doc).to_a }

# Reduce to 2D for visualization
umap = AnnEmbed::UMAP.new(
  n_components: 2,
  n_neighbors: 5,  # Small dataset, use fewer neighbors
  random_seed: 42
)

coords_2d = umap.fit_transform(embeddings)

# Display results
documents.each_with_index do |doc, i|
  x, y = coords_2d[i]
  puts "#{doc[0..30]}... => [#{x.round(3)}, #{y.round(3)}]"
end

# The 2D coordinates should show three distinct clusters
# corresponding to technology, food, and sports topics

# Advanced: Save and load UMAP models for reuse
umap.save("models/document_umap.model")
loaded_umap = AnnEmbed::UMAP.load("models/document_umap.model")
```

## Summary

UMAP is a powerful non-linear dimensionality reduction algorithm particularly suited for visualizing and preprocessing high-dimensional data like embeddings. It excels at preserving both local and global structure through graph-based manifold learning. While it requires sufficient data and careful parameter tuning, it generally outperforms alternatives like t-SNE in speed and quality for most embedding visualization tasks. The key is understanding that UMAP learns a topological representation, not a distance-preserving projection, and interpreting results accordingly.