# frozen_string_literal: true

begin
  # Try to load the compiled extension
  require_relative "annembed_ruby.bundle"
rescue LoadError
  # If that fails, try the .so extension (Linux)
  require_relative "annembed_ruby.so"
end