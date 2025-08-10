# frozen_string_literal: true

module FixtureHelpers
  # Load embedding fixtures from JSON files
  # @param fixture_name [String] Name of the fixture file (without .json extension)
  # @return [Array<Array<Float>>] The embeddings array
  def load_embedding_fixture(fixture_name)
    fixture_path = File.join(File.dirname(__FILE__), '..', 'fixtures', 'embeddings', "#{fixture_name}.json")
    
    unless File.exist?(fixture_path)
      raise "Fixture file not found: #{fixture_path}\n" \
            "Run 'rake fixtures:generate_embeddings' to generate test fixtures."
    end
    
    data = JSON.parse(File.read(fixture_path))
    data['embeddings']
  end
  
  # Load a subset of embeddings from a fixture
  # @param fixture_name [String] Name of the fixture file
  # @param count [Integer] Number of embeddings to load
  # @return [Array<Array<Float>>] Subset of embeddings
  def load_embedding_subset(fixture_name, count)
    embeddings = load_embedding_fixture(fixture_name)
    embeddings.first(count)
  end
  
  # Get metadata about a fixture
  # @param fixture_name [String] Name of the fixture file
  # @return [Hash] Metadata including description, model, dimension, count
  def fixture_metadata(fixture_name)
    fixture_path = File.join(File.dirname(__FILE__), '..', 'fixtures', 'embeddings', "#{fixture_name}.json")
    
    unless File.exist?(fixture_path)
      raise "Fixture file not found: #{fixture_path}"
    end
    
    data = JSON.parse(File.read(fixture_path))
    {
      'description' => data['description'],
      'model' => data['model'],
      'dimension' => data['dimension'],
      'count' => data['count']
    }
  end
  
  # Check if fixtures are available
  # @return [Boolean] true if at least one fixture exists
  def fixtures_available?
    fixtures_dir = File.join(File.dirname(__FILE__), '..', 'fixtures', 'embeddings')
    return false unless Dir.exist?(fixtures_dir)
    
    !Dir.glob(File.join(fixtures_dir, '*.json')).empty?
  end
  
  # Fallback to generate simple structured data if fixtures aren't available
  # This is better than pure random data but not as good as real embeddings
  # @param n_points [Integer] Number of data points
  # @param n_dims [Integer] Number of dimensions
  # @return [Array<Array<Float>>] Generated data with some structure
  def generate_structured_test_data(n_points, n_dims)
    warn "WARNING: Using generated test data instead of real embeddings."
    warn "Run 'rake fixtures:generate_embeddings' for more reliable tests."
    
    # Create data with clusters to avoid degenerate initialization
    n_clusters = [3, n_points / 5].min
    points_per_cluster = n_points / n_clusters
    remainder = n_points % n_clusters
    
    data = []
    n_clusters.times do |c|
      cluster_points = c < remainder ? points_per_cluster + 1 : points_per_cluster
      
      # Each cluster has a different center in the range [-0.1, 0.1]
      center = Array.new(n_dims) { (c.to_f / n_clusters - 0.5) * 0.2 }
      
      cluster_points.times do
        # Add Gaussian-like noise around the center
        point = center.map do |x| 
          # Use Box-Muller transform for better distribution
          theta = 2 * Math::PI * rand
          rho = Math.sqrt(-2 * Math.log(1 - rand))
          noise = rho * Math.cos(theta) * 0.02
          x + noise
        end
        data << point
      end
    end
    
    data
  end
end

# Include in RSpec configuration
RSpec.configure do |config|
  config.include FixtureHelpers
end