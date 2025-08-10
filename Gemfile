source "https://rubygems.org"

# Specify your gem's dependencies in annembed-ruby.gemspec
gemspec

# Test-only dependencies
group :test do
  # Optional: For comparing with Python implementations
  gem "pycall", "~> 1.4", require: false
end

# Development dependencies for generating test fixtures
group :development do
  # For generating real embeddings to use as test fixtures
  # This avoids the hanging issues with random test data
  gem "red-candle", "~> 1.0", require: false
end