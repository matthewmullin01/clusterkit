# Rust Layer Error Handling Documentation

## Overview

The annembed-ruby gem wraps Rust libraries (annembed and hnsw-rs) which have different error handling mechanisms. Some errors can be caught and handled gracefully, while others cause panics that crash the Ruby process.

## Error Categories

### 1. Catchable Errors (Result<T, E> types)

These errors use Rust's `Result` type and can be caught and converted to Ruby exceptions:

| Error | Source | Location | Ruby Exception |
|-------|--------|----------|----------------|
| Isolated point | annembed | `kgraph_from_hnsw_all` | `ClusterKit::IsolatedPointError` |
| Graph construction failure | annembed | `kgraph_from_hnsw_all` | `RuntimeError` with message |
| Embedding failure | annembed | `embedder.embed()` | Generic `RuntimeError` |

**Example from annembed:**
```rust
// This can be caught
return Err(anyhow!(
    "kgraph_from_hnsw_all: graph will not be connected, isolated point at layer {} , pos in layer : {}",
    p_id.0, p_id.1
));
```

**How we handle it in embedder.rs:**
```rust
let kgraph = annembed::fromhnsw::kgraph::kgraph_from_hnsw_all(&hnsw, self.n_neighbors)
    .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
```

### 2. Uncatchable Errors (Panics/Assertions)

These use Rust's `assert!` or `panic!` macros and CANNOT be caught. They will crash the Ruby process:

| Error | Source | Location | Trigger Condition |
|-------|--------|----------|-------------------|
| ~~Box size assertion~~ | ~~annembed~~ | ~~`set_data_box`~~ | **FIXED in cpetersen/annembed:fix-box-size-panic** |
| Array bounds | Various | Index operations | Accessing out-of-bounds indices |
| Unwrap failures | Various | `.unwrap()` calls | Unwrapping `None` or `Err` |

**Update (2025-08-19):** The box size assertion has been fixed in the `fix-box-size-panic` branch of cpetersen/annembed. It now returns a proper `Result<(), anyhow::Error>` that can be caught and handled gracefully:

```rust
// Previously (would panic):
assert!((*f).abs() <= box_size);

// Now (returns catchable error):
if (*f).is_nan() || (*f).is_infinite() {
    return Err(anyhow!("Data normalization failed..."));
}
```

## Current Mitigation Strategies

### 1. Ruby Layer Validation

We validate data before sending to Rust to prevent common panic conditions:

- Check for NaN and Infinite values
- Ensure minimum dataset size (10 points)
- Validate array dimensions consistency
- Warn about extreme value ranges

### 2. Parameter Adjustment

We automatically adjust parameters to avoid error conditions:

```ruby
# Automatically reduce n_neighbors if too large for dataset
adjusted_n_neighbors = [suggested_neighbors, max_neighbors].min
```

### 3. Error Message Enhancement

When we can catch Rust errors, we provide helpful Ruby-level error messages:

```ruby
case error_msg
when /isolated point/i
  raise ::ClusterKit::IsolatedPointError, <<~MSG
    UMAP found isolated points in your data...
    Solutions:
    1. Reduce n_neighbors...
    2. Remove outliers...
  MSG
```

## Previously Uncatchable Panic Conditions (Now Fixed)

### 1. "assertion failed: (*f).abs() <= box_size" - **FIXED**

**Location:** `annembed/src/embedder.rs::set_data_box`

**Previous Issue:** Would panic and crash the Ruby process

**Current Status:** Fixed in `cpetersen/annembed:fix-box-size-panic` branch
- Now returns a catchable `anyhow::Error` 
- Detects NaN/Infinite values during normalization
- Handles constant data (max_max = 0) gracefully
- Extreme value ranges are normalized successfully

**User-visible behavior:** 
- Previously: Ruby process would crash with assertion failure
- Now: Raises a catchable Ruby exception with helpful error message

## Recommendations for Users

### To Avoid Crashes:

1. **Always normalize your data:**
   ```ruby
   # Scale to [0, 1] range
   data = data.map do |row|
     min, max = row.minmax
     range = max - min
     row.map { |v| range > 0 ? (v - min) / range : 0.5 }
   end
   ```

2. **Check for extreme values:**
   ```ruby
   data.flatten.each do |val|
     raise "Extreme value detected" if val.abs > 1e6
   end
   ```

3. **Use conservative parameters for uncertain data:**
   ```ruby
   umap = ClusterKit::Dimensionality::UMAP.new(
     n_neighbors: 5,  # Lower is safer
     n_components: 2
   )
   ```

## Future Improvements

### Potential Solutions:

1. **Modify annembed to use Result instead of assert:**
   - Would require upstream changes to annembed
   - Convert `assert!` to `if` checks that return `Err`

2. **Add panic catching in Rust layer:**
   - Use `std::panic::catch_unwind` (limited effectiveness)
   - May not work for all panic types

3. **Pre-validation in Rust:**
   - Add more checks before calling annembed functions
   - Predict and prevent panic conditions

### Current Limitations:

- Cannot catch Rust panics from Ruby
- Some numerical instabilities are hard to predict
- Trade-off between performance and safety checks

## Testing Error Handling

The test suite mocks Rust errors to verify our error handling logic works correctly. However, actual panic conditions cannot be tested without crashing the test process.

See `spec/clusterkit/error_handling_spec.rb` for error handling tests.