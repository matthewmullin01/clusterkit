# CLAUDE.md - clusterkit Project Guide

## Project Vision
clusterkit brings high-performance dimensionality reduction and embedding algorithms to Ruby by wrapping the annembed Rust crate. This gem is part of the ruby-nlp ecosystem, which aims to provide Ruby developers with native machine learning and NLP capabilities through best-in-breed Rust implementations.

## Core Principles

### 1. Ruby-First Design
- Provide an idiomatic Ruby API that feels natural to Ruby developers
- Follow Ruby naming conventions (snake_case methods, proper use of symbols)
- Support Ruby's duck typing while maintaining type safety at the Rust boundary
- Integrate seamlessly with Ruby's data science ecosystem

### 2. Performance Without Compromise
- Leverage Rust's performance for compute-intensive operations
- Use Magnus for zero-copy data transfer where possible
- Enable parallelization by default
- Provide progress feedback for long-running operations

### 3. Ecosystem Integration
- Primary support for Numo::NArray (the NumPy of Ruby)
- Work well with other ruby-nlp gems (lancelot, red-candle)
- Support common Ruby data formats and visualization tools
- Play nice with Jupyter notebooks (iruby)

## Technical Guidelines

### Magnus Best Practices

1. **Memory Management**
   ```rust
   // Good: Let Magnus handle Ruby object lifecycle
   let array: RArray = data.try_convert()?;
   
   // Avoid: Manual memory management
   // Don't try to manually free Ruby objects
   ```

2. **Error Handling**
   ```rust
   // Always wrap errors properly
   use magnus::Error;
   
   fn risky_operation() -> Result<RArray, Error> {
       annembed_call()
           .map_err(|e| Error::new(exception::runtime_error(), e.to_string()))?
   }
   ```

3. **Type Conversions**
   ```rust
   // Define clear conversion traits
   impl TryFrom<Value> for EmbedConfig {
       type Error = Error;
       // Robust conversion with good error messages
   }
   ```

### Ruby API Design

1. **Method Naming**
   - Use Ruby conventions: `fit_transform`, not `fitTransform`
   - Predicates end with `?`: `converged?`, `fitted?`
   - Dangerous methods end with `!`: `normalize!`

2. **Parameter Handling**
   ```ruby
   # Good: Use keyword arguments with defaults
   def initialize(method: :umap, n_components: 2, **options)
   
   # Avoid: Positional arguments for configuration
   def initialize(method, n_components, min_dist, spread, ...)
   ```

3. **Return Values**
   - Return Ruby arrays for small results
   - Return Numo::NArray for large matrices
   - Support multiple return formats via options

### Performance Considerations

1. **Data Transfer**
   - Minimize copying between Ruby and Rust
   - Use view/slice operations when possible
   - Support streaming for large datasets

2. **Threading**
   - Respect Ruby's GVL (Global VM Lock)
   - Release GVL for long-running Rust operations
   - Use Rust's parallelization, not Ruby threads

3. **Memory Usage**
   - Provide memory estimates for large operations
   - Support out-of-core processing for huge datasets
   - Clear progress indication for long operations

## Code Style Guidelines

### Rust Side
- Follow Rust standard style (rustfmt)
- Comprehensive error types with context
- Document all public functions
- Use type aliases for clarity

### Ruby Side
- Follow Ruby Style Guide
- Use YARD documentation format
- Provide type signatures where helpful
- Include usage examples in docs

## Testing Philosophy

1. **Comprehensive Coverage**
   - Unit tests for all public methods
   - Integration tests with real datasets
   - Performance benchmarks
   - Memory leak tests

2. **Test Data**
   - Use standard ML datasets (Iris, MNIST samples)
   - Generate synthetic data for edge cases
   - Test with various Ruby object types

3. **Platform Testing**
   - Test on multiple Ruby versions
   - Test on different operating systems
   - Verify precompiled gem distribution

## Documentation Standards

1. **README**
   - Clear installation instructions
   - Quick start example that works
   - Link to full documentation
   - Performance comparisons

2. **API Documentation**
   - Every public method documented
   - Parameter types and ranges specified
   - Return values clearly described
   - Usage examples for complex methods

3. **Tutorials**
   - Jupyter notebook examples
   - Common use case walkthroughs
   - Integration examples with other gems

## Common Patterns

### Configuration Objects
```ruby
# Prefer configuration objects over many parameters
config = Annembed::Config.new(
  method: :umap,
  n_neighbors: 15,
  min_dist: 0.1
)
embedder = Annembed::Embedder.new(config)
```

### Progress Callbacks
```ruby
# Support progress monitoring
embedder.on_progress do |iteration, total|
  puts "Progress: #{iteration}/#{total}"
end
```

### Flexible Input/Output
```ruby
# Accept multiple input formats
embedder.fit_transform(data)  # Array, NArray, or CSV path

# Support different output formats
result = embedder.transform(data, output: :array)    # Ruby Array
result = embedder.transform(data, output: :narray)   # Numo::NArray
```

## Development Workflow

1. **Branch Strategy**
   - `main` - stable release
   - `develop` - integration branch
   - `feature/*` - new features
   - `fix/*` - bug fixes

2. **Release Process**
   - Version bump in version.rb
   - Update CHANGELOG.md
   - Run full test suite
   - Build precompiled gems
   - Tag release
   - Push to RubyGems

3. **Continuous Integration**
   - Run tests on each push
   - Build gems for multiple platforms
   - Check documentation building
   - Performance regression tests

## Future Considerations

1. **GPU Support**
   - Monitor annembed for GPU features
   - Plan bindings if GPU support is added
   - Consider alternative GPU libraries

2. **Web Integration**
   - Consider Rails integration
   - WebAssembly compilation?
   - REST API wrapper?

3. **Visualization**
   - Built-in plotting helpers?
   - Export to common formats
   - Interactive visualizations?

## Getting Help

When implementing new features:
1. Check existing patterns in lancelot and red-candle
2. Consult annembed documentation
3. Ask in ruby-nlp discussions
4. Profile before optimizing

Remember: The goal is to make advanced embedding algorithms accessible and performant for Ruby developers while maintaining the simplicity and elegance that makes Ruby special.