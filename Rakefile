require "bundler/gem_tasks"
require "rake/extensiontask"
require "rspec/core/rake_task"

# RSpec test task
RSpec::Core::RakeTask.new(:spec)

# Define the Rust extension
spec = Gem::Specification.load("annembed-ruby.gemspec")
Rake::ExtensionTask.new("annembed_ruby", spec) do |ext|
  ext.lib_dir = "lib/annembed"
  ext.source_pattern = "*.{rs,toml}"
  ext.cross_compile = true
  ext.cross_platform = %w[x86-mingw32 x64-mingw32 x86-linux x86_64-linux x86_64-darwin arm64-darwin]
end

task default: [:compile, :spec]

# Documentation task
begin
  require "yard"
  YARD::Rake::YardocTask.new do |t|
    t.files = ["lib/**/*.rb"]
    t.options = ["--no-private", "--readme", "README.md"]
  end
rescue LoadError
  desc "YARD documentation task not available"
  task :yard do
    puts "YARD is not available. Please install it with: gem install yard"
  end
end

# Benchmarking task
desc "Run benchmarks"
task :benchmark do
  ruby "test/benchmark/benchmarks.rb"
end

# Console task for interactive testing
desc "Open an interactive console with the gem loaded"
task :console do
  require "irb"
  require "annembed"
  ARGV.clear
  IRB.start
end

# Rust-specific tasks
namespace :rust do
  desc "Run cargo fmt"
  task :fmt do
    Dir.chdir("ext/annembed_ruby") do
      sh "cargo fmt"
    end
  end

  desc "Run cargo clippy"
  task :clippy do
    Dir.chdir("ext/annembed_ruby") do
      sh "cargo clippy -- -D warnings"
    end
  end

  desc "Run cargo test"
  task :test do
    Dir.chdir("ext/annembed_ruby") do
      sh "cargo test"
    end
  end
end

# CI task that runs all checks
desc "Run all CI checks"
task ci: ["rust:fmt", "rust:clippy", "compile", "spec"]