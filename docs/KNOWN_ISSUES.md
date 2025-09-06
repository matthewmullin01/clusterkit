# Known Issues

## Summary

This gem has three main categories of limitations:

1. **Minimum dataset requirements** - UMAP needs at least 10 data points
2. **Performance trade-offs** - Reproducibility (with seed) is ~25-35% slower than parallel mode
3. **Uncatchable Rust panics** - Some error conditions crash the Ruby process (cannot be caught)

## Minimum Dataset Size Requirement

**Limitation**: UMAP requires at least 10 data points to function properly.

**Reason**: UMAP needs sufficient data to construct a meaningful manifold approximation. With fewer than 10 points, the algorithm cannot create a reliable graph structure.

**Workaround**:
- Use PCA for datasets with fewer than 10 points
- The `transform` method can handle smaller datasets once the model is fitted on adequate training data

## Performance vs Reproducibility Trade-off

**Design Choice**: When using `random_seed` for reproducibility, UMAP uses serial processing which is approximately 25-35% slower than parallel processing.

**Recommendation**:
- For production workloads where speed is critical: omit the `random_seed` parameter
- For research, testing, or when reproducibility is required: provide a `random_seed` value

## Rust Panic Conditions (Mostly Fixed)

**Previous Issue**: The box_size assertion would panic and crash the Ruby process.

**Current Status**: **FIXED** in `scientist-labs/annembed:fix-box-size-panic` branch
- The `"assertion failed: (*f).abs() <= box_size"` panic has been converted to a catchable error
- Extreme value ranges are now handled gracefully through normalization
- NaN/Infinite values are detected and reported with clear error messages

**Remaining Uncatchable Errors**:
- Array bounds violations (accessing out-of-bounds indices)
- Some `.unwrap()` calls on `None` or `Err` values
- These are much less common in normal usage

**Best Practices** (still recommended):
- Normalize your data to a reasonable range (e.g., 0-1) for best performance
- Remove or handle NaN/Infinite values before processing
- Use conservative parameters when data quality is uncertain

**For more details**: See [RUST_ERROR_HANDLING.md](RUST_ERROR_HANDLING.md) for comprehensive documentation of error handling limitations.

## Best Practices to Avoid Issues

### Data Preprocessing

Always preprocess your data before using UMAP:

```ruby
# 1. Remove NaN and Infinite values
data.reject! { |row| row.any? { |v| v.nan? || v.infinite? } }

# 2. Normalize to [0, 1] range
data = data.map do |row|
  min, max = row.minmax
  range = max - min
  row.map { |v| range > 0 ? (v - min) / range : 0.5 }
end

# 3. Check for extreme outliers
data.each do |row|
  row.each do |val|
    if val.abs > 100
      warn "Warning: Extreme value #{val} detected"
    end
  end
end
```

### Safe Parameter Defaults

Use conservative parameters when data quality is uncertain:

```ruby
# Safer configuration
umap = ClusterKit::Dimensionality::UMAP.new(
  n_components: 2,
  n_neighbors: 5,        # Lower is safer (default: 15)
  random_seed: 42,       # For reproducibility during debugging
  nb_grad_batch: 10,     # Default is usually fine
  nb_sampling_by_edge: 8 # Default is usually fine
)
```

### Error Recovery Strategy

Since some errors cannot be caught, implement a recovery strategy:

```ruby
def safe_umap_transform(data, options = {})
  # Save data to temporary file before processing
  temp_file = "temp_umap_data_#{Time.now.to_i}.json"
  File.write(temp_file, JSON.dump(data))

  begin
    umap = ClusterKit::Dimensionality::UMAP.new(**options)
    result = umap.fit_transform(data)
    File.delete(temp_file) if File.exist?(temp_file)
    result
  rescue => e
    puts "UMAP failed: #{e.message}"
    puts "Data saved to #{temp_file} for debugging"
    raise
  end
end
```

### Alternative for Problematic Data

If UMAP consistently fails, use PCA as a fallback:

```ruby
def reduce_dimensions(data, n_components: 2)
  begin
    umap = ClusterKit::Dimensionality::UMAP.new(n_components: n_components)
    umap.fit_transform(data)
  rescue => e
    warn "UMAP failed, falling back to PCA: #{e.message}"
    pca = ClusterKit::Dimensionality::PCA.new(n_components: n_components)
    pca.fit_transform(data)
  end
end
```
