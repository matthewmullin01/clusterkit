# Implementation Notes for annembed-ruby

## Architecture Overview

The gem follows a three-layer architecture:

1. **Ruby Layer** (`lib/annembed/`): User-facing API with Ruby idioms
2. **Magnus Bridge** (`ext/annembed_ruby/src/`): Type conversions and bindings
3. **annembed Core**: The underlying Rust embedding library

## Key Implementation Challenges

### 1. Data Type Conversions

The main challenge is efficiently converting between Ruby and Rust types:

```
Ruby Array/Numo::NArray ↔ ndarray::Array2<f32> ↔ annembed matrices
```

Consider using views instead of copies where possible.

### 2. Memory Management

- Magnus handles Ruby GC integration
- Need to be careful with large matrices
- Consider streaming for datasets > available RAM

### 3. Progress Reporting

annembed operations can be long-running. Options:
1. Polling from Ruby side (requires thread)
2. Callback from Rust (needs GVL management)
3. Async/await pattern (complex but clean)

### 4. Error Handling

Map annembed errors to Ruby exceptions:
- `annembed::Error` → `Annembed::Error`
- Panic → `RuntimeError` (avoid panics!)
- Invalid input → `ArgumentError`

## annembed API Mapping

### Core Types
- `EmbedderT` - Main embedding trait
- `HnswParams` - HNSW graph parameters
- `EmbedderParams` - Algorithm-specific params
- `GraphProjection` - For graph-based methods

### Key Functions
```rust
// Main embedding function
pub fn get_embedder(
    data: &Array2<f32>,
    params: EmbedderParams,
    hnsw_params: HnswParams
) -> Result<Box<dyn EmbedderT>>

// The embedder trait
pub trait EmbedderT {
    fn embed(&self) -> Result<Array2<f32>>;
    fn get_hnsw(&self) -> &Hnsw<f32, DistL2>;
}
```

## Performance Considerations

1. **Parallelization**: annembed uses rayon, respect Ruby's thread settings
2. **BLAS Backend**: Allow users to choose (OpenBLAS vs MKL)
3. **Large Datasets**: Implement chunking for > 1M points
4. **GPU Future**: Design API to allow GPU backend later

## Testing Strategy

### Unit Tests
- Type conversions
- Configuration parsing
- Error handling

### Integration Tests
- Small datasets (Iris)
- Medium datasets (MNIST subset)
- Large datasets (if CI allows)

### Benchmarks
- vs Python UMAP
- vs pure Ruby implementations
- Memory usage profiling

## Platform Support

### Priorities
1. Linux x86_64 (most servers)
2. macOS arm64 (M1/M2 developers)
3. macOS x86_64 (Intel Macs)
4. Windows x86_64 (if feasible)

### Build Considerations
- Use rake-compiler-dock for cross-compilation
- Static link BLAS when possible
- Provide clear instructions for source builds

## Future Enhancements

1. **Incremental Learning**: Add new points to existing embedding
2. **Custom Metrics**: Allow user-defined distance functions
3. **Supervised Embedding**: Use labels to guide embedding
4. **GPU Support**: If annembed adds it
5. **Visualization**: Built-in plotting helpers?

## Debugging Tips

1. Use `RUST_BACKTRACE=1` for better errors
2. Add logging with `env_logger` in Rust
3. Use `rb_sys` debug mode for development
4. Memory debugging with valgrind/ASAN

## Code Organization

Keep related functionality together:
- `embedder.rs`: All embedding algorithms
- `utils.rs`: Dimension estimation, hubness
- `svd.rs`: Matrix decomposition
- `conversions.rs`: Type conversion helpers

## Release Checklist

1. [ ] Update version in `version.rb`
2. [ ] Update CHANGELOG.md
3. [ ] Run full test suite on all platforms
4. [ ] Build precompiled gems
5. [ ] Test gem installation from .gem file
6. [ ] Tag release in git
7. [ ] Push to RubyGems.org
8. [ ] Update documentation

## References

- annembed docs: https://docs.rs/annembed/
- Magnus guide: https://github.com/matsadler/magnus
- UMAP paper: https://arxiv.org/abs/1802.03426
- t-SNE paper: https://lvdmaaten.github.io/tsne/