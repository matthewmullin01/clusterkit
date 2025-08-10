# Test Fixtures for UMAP Testing

## Overview

To avoid the hanging issues that occur when testing UMAP with synthetic random data, we use real embeddings from text models as test fixtures. This ensures our tests are both reliable and realistic.

## Why Real Embeddings?

UMAP's initialization algorithm (`dmap_init`) expects data with manifold structure - the kind of structure that real embeddings naturally have. When given uniform random data, it can fail catastrophically, initializing all points to the same location and causing infinite loops.

Real text embeddings have:
- Natural clustering (semantically similar texts group together)
- Meaningful correlations between dimensions
- Appropriate value ranges (typically [-0.12, 0.12])
- Inherent manifold structure that UMAP is designed to discover

## Generating Fixtures

### Prerequisites

1. Install the development dependencies:
   ```bash
   bundle install --with development
   ```

2. Generate the embedding fixtures:
   ```bash
   rake fixtures:generate_embeddings
   ```

This will create several fixture files in `spec/fixtures/embeddings/`:

- **basic_15.json** - 15 general sentences for basic testing
- **clusters_30.json** - 30 sentences in 3 distinct topic clusters (tech, nature, food)
- **minimal_6.json** - 6 sentences for minimum viable dataset testing
- **large_100.json** - 100 sentences for performance testing

### Fixture Format

Each fixture is a JSON file containing:
```json
{
  "description": "Test embeddings for basic_15",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimension": 384,
  "count": 15,
  "embeddings": [
    [0.123, -0.045, ...],  // 384-dimensional vectors
    ...
  ]
}
```

## Using Fixtures in Tests

The specs automatically use fixtures when available:

```ruby
RSpec.describe "My UMAP test" do
  let(:test_data) do
    if fixtures_available?
      load_embedding_fixture('basic_15')  # Real embeddings
    else
      generate_structured_test_data(15, 30)  # Fallback
    end
  end
  
  it "processes embeddings" do
    umap = AnnEmbed::UMAP.new
    result = umap.fit_transform(test_data)
    # Test will use real embeddings, avoiding hanging issues
  end
end
```

### Available Helper Methods

- `fixtures_available?` - Check if any fixtures exist
- `load_embedding_fixture(name)` - Load all embeddings from a fixture
- `load_embedding_subset(name, count)` - Load first N embeddings
- `fixture_metadata(name)` - Get metadata about a fixture
- `generate_structured_test_data(n_points, n_dims)` - Fallback data generator

## Listing Available Fixtures

To see what fixtures are available:
```bash
rake fixtures:list
```

Output:
```
Available embedding fixtures:
  basic_15.json: 15 embeddings, 384D
  clusters_30.json: 30 embeddings, 384D
  minimal_6.json: 6 embeddings, 384D
  large_100.json: 100 embeddings, 384D
```

## When to Regenerate Fixtures

Regenerate fixtures when:
- Switching to a different embedding model
- Adding new test scenarios
- Fixtures become corrupted or deleted

The fixtures are deterministic for a given model and input text, so regenerating them should produce functionally equivalent embeddings.

## CI/CD Considerations

For CI environments, you have two options:

1. **Commit fixtures to git** (Recommended for small fixtures):
   ```bash
   git add spec/fixtures/embeddings/*.json
   git commit -m "Add embedding test fixtures"
   ```

2. **Generate fixtures in CI**:
   Add to your CI workflow:
   ```yaml
   - name: Generate test fixtures
     run: bundle exec rake fixtures:generate_embeddings
   ```

Note: Option 2 requires red-candle to be available in CI, which will download the embedding model on first use.

## Troubleshooting

### "Fixture file not found" Error
Run `rake fixtures:generate_embeddings` to create the fixtures.

### Tests Still Hanging
Ensure you're using the fixture data, not generating random data. Check that your test includes:
```ruby
if fixtures_available?
  load_embedding_fixture('basic_15')
```

### red-candle Not Found
Install development dependencies:
```bash
bundle install --with development
```

### Model Download Issues
The first run will download the embedding model (~90MB). Ensure you have internet connectivity and sufficient disk space.

## Adding New Fixtures

To add new test scenarios, edit `Rakefile` and add to the `test_cases` hash:

```ruby
test_cases = {
  'my_new_test' => [
    "First test sentence",
    "Second test sentence",
    # ...
  ]
}
```

Then regenerate:
```bash
rake fixtures:generate_embeddings
```

## Performance Note

Using real embeddings makes tests slightly slower than random data, but the reliability improvement is worth it. The fixtures are loaded from JSON, which is fast, and the UMAP algorithm actually converges properly instead of hanging.