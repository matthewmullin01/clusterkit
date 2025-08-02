# frozen_string_literal: true

# Helper to suppress output at the C level
module OutputSuppressor
  def self.suppress_output
    if RUBY_PLATFORM =~ /darwin|linux/
      # Use Unix file descriptors to redirect stdout/stderr
      begin
        # Save original stdout/stderr
        orig_stdout = STDOUT.dup
        orig_stderr = STDERR.dup
        
        # Redirect to /dev/null
        STDOUT.reopen('/dev/null', 'w')
        STDERR.reopen('/dev/null', 'w')
        
        yield
      ensure
        # Restore original stdout/stderr
        STDOUT.reopen(orig_stdout)
        STDERR.reopen(orig_stderr)
        orig_stdout.close
        orig_stderr.close
      end
    else
      # On Windows or if the above doesn't work, just yield
      yield
    end
  end
end