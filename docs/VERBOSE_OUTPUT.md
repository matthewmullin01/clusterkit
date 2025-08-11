# Controlling Verbose Output

The annembed-ruby gem provides control over the verbose debug output from the underlying Rust library.

## Default Behavior

By default, annembed-ruby suppresses the debug output from the Rust library to keep your console clean. This includes messages about quantiles, cross entropy values, and gradient iterations.

## Enabling Verbose Output

There are two ways to enable verbose output:

### 1. Environment Variable

Set the `ANNEMBED_VERBOSE` environment variable:

```bash
ANNEMBED_VERBOSE=true ruby your_script.rb
```

Or in your Ruby code:

```ruby
ENV['ANNEMBED_VERBOSE'] = 'true'
require 'annembed'
```

### 2. Configuration API

Use the configuration API for programmatic control:

```ruby
require 'annembed'

# Enable verbose output
AnnEmbed.configure do |config|
  config.verbose = true
end

# Your UMAP operations will now show debug output
umap = AnnEmbed::UMAP.new
umap.fit_transform(data)

# Disable verbose output
AnnEmbed.configuration.verbose = false
```

## When to Use Verbose Output

Verbose output is useful for:

- **Debugging convergence issues**: See iteration counts and cross entropy values
- **Understanding performance**: Monitor gradient descent progress
- **Troubleshooting edge cases**: Identify degenerate initializations or disconnected graphs
- **Development and testing**: Verify the algorithm is working correctly

## Example Output

When verbose mode is enabled, you'll see output like:

```
constructed initial space

scales quantile at 0.05 : 1.12e0 , 0.5 :  1.17e0, 0.95 : 1.23e0, 0.99 : 1.23e0

edge weight quantile at 0.05 : 1.87e-1 , 0.5 :  1.99e-1, 0.95 : 2.15e-1, 0.99 : 2.19e-1

perplexity quantile at 0.05 : 4.99e0 , 0.5 :  5.00e0, 0.95 : 5.00e0, 0.99 : 5.00e0

embedded scales quantiles at 0.05 : 1.91e-1 , 0.5 :  2.00e-1, 0.95 : 2.10e-1, 0.99 : 2.10e-1

initial cross entropy value 7.48e1,  in time 972µs
gradient iterations sys time(s) 0.00e0 , cpu_time(s) 0.00e0
final cross entropy value 5.85e1
```

## Implementation Details

The gem uses Ruby's standard approach for suppressing output from C/Rust extensions:
- Output streams are temporarily redirected to `/dev/null` (Unix) or `NUL:` (Windows)
- The redirection happens at the file descriptor level to capture C/Rust `printf`/`println!` output
- After the operation completes, streams are restored to their original state

This approach is thread-safe within the context of Ruby's Global VM Lock (GVL) and follows patterns used by popular Ruby gems like Rails/ActiveSupport.