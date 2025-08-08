# frozen_string_literal: true

module AnnEmbed
  # Configuration class for embedding algorithms
  class Config
    # HNSW parameters
    attr_accessor :ef_construction, :max_nb_connection, :nb_layer
    
    # Common embedding parameters
    attr_accessor :n_components, :n_neighbors, :random_seed, :n_threads
    
    # UMAP specific
    attr_accessor :min_dist, :spread, :local_connectivity, :set_op_mix_ratio
    attr_accessor :negative_sample_rate, :transform_queue_size
    
    # t-SNE specific
    attr_accessor :perplexity, :learning_rate, :n_iter, :early_exaggeration
    attr_accessor :theta, :eta
    
    # Algorithm selection
    attr_accessor :method
    
    # Initialize configuration with defaults
    def initialize(method: :umap, **options)
      @method = method
      
      # Set defaults based on method
      set_defaults(method)
      
      # Override with user options
      options.each do |key, value|
        if respond_to?("#{key}=")
          send("#{key}=", value)
        else
          warn "Unknown option: #{key}"
        end
      end
    end

    # Convert to hash for passing to Rust
    def to_h
      instance_variables.each_with_object({}) do |var, hash|
        key = var.to_s.delete_prefix("@").to_sym
        hash[key] = instance_variable_get(var)
      end
    end

    private

    def set_defaults(method)
      # Common defaults
      @n_components = 2
      @random_seed = nil
      @n_threads = nil # Use all available
      
      # HNSW defaults
      @ef_construction = 200
      @max_nb_connection = 16
      @nb_layer = 16
      
      case method
      when :umap
        set_umap_defaults
      when :tsne
        set_tsne_defaults
      when :largevis
        set_largevis_defaults
      when :diffusion
        set_diffusion_defaults
      end
    end

    def set_umap_defaults
      @n_neighbors = 15
      @min_dist = 0.1
      @spread = 1.0
      @local_connectivity = 1.0
      @set_op_mix_ratio = 1.0
      @negative_sample_rate = 5
      @transform_queue_size = 4.0
    end

    def set_tsne_defaults
      @perplexity = 30.0
      @learning_rate = 200.0
      @n_iter = 1000
      @early_exaggeration = 12.0
      @theta = 0.5
      @eta = 200.0
    end

    def set_largevis_defaults
      @n_neighbors = 15
      @perplexity = 30.0
      @learning_rate = 1.0
      @n_iter = 1000
    end

    def set_diffusion_defaults
      @n_neighbors = 15
      @alpha = 1.0
      @n_iter = 1
    end
  end
end