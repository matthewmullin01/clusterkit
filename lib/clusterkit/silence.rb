# frozen_string_literal: true

module ClusterKit
  # Module to suppress stdout output from the Rust library
  # Following the pattern used by Rails/ActiveSupport and other popular gems
  module Silence
    # Temporarily silence stdout and stderr
    # This is the most idiomatic Ruby way to suppress output from C/Rust extensions
    # 
    # @example
    #   ClusterKit::Silence.silence_stream(STDOUT) do
    #     # code that produces unwanted output
    #   end
    def self.silence_stream(stream)
      old_stream = stream.dup
      stream.reopen(File::NULL)
      stream.sync = true
      yield
    ensure
      stream.reopen(old_stream)
      old_stream.close
    end

    # Silence both stdout and stderr
    def self.silence_output
      silence_stream(STDOUT) do
        silence_stream(STDERR) do
          yield
        end
      end
    end

    # Conditionally silence based on configuration
    def self.maybe_silence
      if AnnEmbed.configuration.verbose
        yield
      else
        silence_output { yield }
      end
    end
  end
end