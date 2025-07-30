source "https://rubygems.org"

# Specify your gem's dependencies in annembed-ruby.gemspec
gemspec

group :development do
  gem "rake", "~> 13.0"
  gem "rake-compiler", "~> 1.2"
  gem "minitest", "~> 5.0"
  gem "minitest-reporters", "~> 1.6"
  gem "yard", "~> 0.9"
end

group :test do
  # Optional: For comparing with Python implementations
  gem "pycall", "~> 1.4", require: false
  
  # For matrix operations in tests
  gem "numo-narray", "~> 0.9"
end