# frozen_string_literal: true

begin
  # Try to load the compiled extension
  require_relative "clusterkit.bundle"
rescue LoadError
  # If that fails, try the .so extension (Linux)
  require_relative "clusterkit.so"
end