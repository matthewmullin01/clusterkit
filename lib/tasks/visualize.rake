namespace :annembed do
  desc "Generate interactive visualization comparing dimensionality reduction and clustering methods"
  task :visualize, [:output_file, :dataset, :clustering] do |t, args|
    require 'bundler/setup'
    require 'annembed'
    require 'json'
    
    output_file = args[:output_file] || 'annembed_visualization.html'
    dataset_type = args[:dataset] || 'clusters'
    clustering_method = args[:clustering] || 'both'  # 'kmeans', 'hdbscan', or 'both'
    
    puts "Generating visualization with dataset: #{dataset_type}, clustering: #{clustering_method}"
    
    # Generate dataset based on type
    data, true_labels, dataset_name = case dataset_type
    when 'swiss'
      generate_swiss_roll
    when 'iris'
      generate_iris_like_data
    else
      generate_clustered_data
    end
    
    puts "Generated #{data.size} points in #{data.first.size} dimensions"
    
    # Reduce dimensions
    print "Running UMAP..."
    umap = AnnEmbed::Embedder.new(method: :umap, n_components: 2, n_neighbors: 15, random_seed: 42)
    umap_data = umap.fit_transform(data)
    puts " done"
    
    # Create 20D UMAP for HDBSCAN (better for density-based clustering)
    print "Running UMAP to 20D for HDBSCAN..."
    umap_20d = AnnEmbed::UMAP.new(n_components: 20, n_neighbors: 15)
    umap_data_20d = umap_20d.fit_transform(data)
    puts " done"
    
    print "Running PCA..."
    pca = AnnEmbed::PCA.new(n_components: 2)
    pca_data = pca.fit_transform(data)
    variance_explained = pca.cumulative_explained_variance_ratio[-1]
    puts " done (explained variance: #{(variance_explained * 100).round(1)}%)"
    
    print "Running SVD..."
    u, s, vt = AnnEmbed.svd(data, 2, n_iter: 5)
    svd_data = u
    puts " done"
    
    # Initialize clustering results
    clustering_results = {}
    metrics = {
      pca_variance_explained: variance_explained
    }
    
    # Perform K-means clustering if requested
    if clustering_method == 'kmeans' || clustering_method == 'both'
      print "Clustering with K-means..."
      
      # Find optimal k using elbow method
      elbow_results = AnnEmbed::Clustering.elbow_method(umap_data, k_range: 2..6)
      
      # Use library method to detect optimal k
      optimal_k = AnnEmbed::Clustering.detect_optimal_k(elbow_results)
      
      puts "\n  Elbow method results:"
      elbow_results.sort.each do |k, inertia|
        puts "    k=#{k}: #{inertia.round(2)}"
      end
      puts "  Detected optimal k: #{optimal_k}"
      
      kmeans_umap = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
      kmeans_labels_umap = kmeans_umap.fit_predict(umap_data)
      
      kmeans_pca = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
      kmeans_labels_pca = kmeans_pca.fit_predict(pca_data)
      
      kmeans_svd = AnnEmbed::Clustering::KMeans.new(k: optimal_k, random_seed: 42)
      kmeans_labels_svd = kmeans_svd.fit_predict(svd_data)
      
      # Calculate K-means metrics
      silhouette_umap_kmeans = AnnEmbed::Clustering.silhouette_score(umap_data, kmeans_labels_umap)
      silhouette_pca_kmeans = AnnEmbed::Clustering.silhouette_score(pca_data, kmeans_labels_pca)
      silhouette_svd_kmeans = AnnEmbed::Clustering.silhouette_score(svd_data, kmeans_labels_svd)
      
      clustering_results[:kmeans] = {
        labels_umap: kmeans_labels_umap,
        labels_pca: kmeans_labels_pca,
        labels_svd: kmeans_labels_svd,
        optimal_k: optimal_k,
        elbow_results: elbow_results
      }
      
      metrics[:kmeans] = {
        silhouette_umap: silhouette_umap_kmeans,
        silhouette_pca: silhouette_pca_kmeans,
        silhouette_svd: silhouette_svd_kmeans,
        optimal_k: optimal_k
      }
      
      puts " done"
    end
    
    # Perform HDBSCAN clustering if requested
    if clustering_method == 'hdbscan' || clustering_method == 'both'
      print "Clustering with HDBSCAN..."
      
      # HDBSCAN on 20D UMAP (better for density-based clustering)
      hdbscan = AnnEmbed::Clustering::HDBSCAN.new(
        min_samples: 5,
        min_cluster_size: 10
      )
      hdbscan_labels_20d = hdbscan.fit_predict(umap_data_20d)
      
      # For visualization consistency, also cluster the 2D projections
      hdbscan_2d = AnnEmbed::Clustering::HDBSCAN.new(
        min_samples: 5,
        min_cluster_size: 10
      )
      hdbscan_labels_umap = hdbscan_2d.fit_predict(umap_data)
      
      hdbscan_pca = AnnEmbed::Clustering::HDBSCAN.new(
        min_samples: 5,
        min_cluster_size: 10
      )
      hdbscan_labels_pca = hdbscan_pca.fit_predict(pca_data)
      
      hdbscan_svd = AnnEmbed::Clustering::HDBSCAN.new(
        min_samples: 5,
        min_cluster_size: 10
      )
      hdbscan_labels_svd = hdbscan_svd.fit_predict(svd_data)
      
      puts "\n  HDBSCAN results (20D):"
      puts "    Clusters found: #{hdbscan.n_clusters}"
      puts "    Noise points: #{hdbscan.n_noise_points} (#{(hdbscan.noise_ratio * 100).round(1)}%)"
      
      # Calculate HDBSCAN metrics (excluding noise for silhouette)
      non_noise_mask_umap = hdbscan_labels_umap.map { |l| l != -1 }
      non_noise_mask_pca = hdbscan_labels_pca.map { |l| l != -1 }
      non_noise_mask_svd = hdbscan_labels_svd.map { |l| l != -1 }
      
      # Filter out noise points for silhouette calculation
      if non_noise_mask_umap.any? { |m| m }
        filtered_data_umap = umap_data.select.with_index { |_, i| non_noise_mask_umap[i] }
        filtered_labels_umap = hdbscan_labels_umap.select.with_index { |l, i| non_noise_mask_umap[i] }
        silhouette_umap_hdbscan = filtered_labels_umap.uniq.size > 1 ? 
          AnnEmbed::Clustering.silhouette_score(filtered_data_umap, filtered_labels_umap) : 0.0
      else
        silhouette_umap_hdbscan = 0.0
      end
      
      if non_noise_mask_pca.any? { |m| m }
        filtered_data_pca = pca_data.select.with_index { |_, i| non_noise_mask_pca[i] }
        filtered_labels_pca = hdbscan_labels_pca.select.with_index { |l, i| non_noise_mask_pca[i] }
        silhouette_pca_hdbscan = filtered_labels_pca.uniq.size > 1 ?
          AnnEmbed::Clustering.silhouette_score(filtered_data_pca, filtered_labels_pca) : 0.0
      else
        silhouette_pca_hdbscan = 0.0
      end
      
      if non_noise_mask_svd.any? { |m| m }
        filtered_data_svd = svd_data.select.with_index { |_, i| non_noise_mask_svd[i] }
        filtered_labels_svd = hdbscan_labels_svd.select.with_index { |l, i| non_noise_mask_svd[i] }
        silhouette_svd_hdbscan = filtered_labels_svd.uniq.size > 1 ?
          AnnEmbed::Clustering.silhouette_score(filtered_data_svd, filtered_labels_svd) : 0.0
      else
        silhouette_svd_hdbscan = 0.0
      end
      
      clustering_results[:hdbscan] = {
        labels_umap: hdbscan_labels_umap,
        labels_pca: hdbscan_labels_pca,
        labels_svd: hdbscan_labels_svd,
        labels_20d: hdbscan_labels_20d,  # The main HDBSCAN result
        n_clusters: hdbscan.n_clusters,
        n_noise: hdbscan.n_noise_points,
        noise_ratio: hdbscan.noise_ratio
      }
      
      metrics[:hdbscan] = {
        silhouette_umap: silhouette_umap_hdbscan,
        silhouette_pca: silhouette_pca_hdbscan,
        silhouette_svd: silhouette_svd_hdbscan,
        n_clusters: hdbscan.n_clusters,
        noise_ratio: hdbscan.noise_ratio
      }
      
      puts " done"
    end
    
    # Generate HTML
    html = generate_visualization_html(
      data: data,
      umap_data: umap_data,
      pca_data: pca_data,
      svd_data: svd_data,
      true_labels: true_labels,
      clustering_results: clustering_results,
      dataset_name: dataset_name,
      metrics: metrics,
      clustering_method: clustering_method
    )
    
    File.write(output_file, html)
    puts "\nVisualization saved to: #{output_file}"
    puts "Open in browser: open #{output_file}"
  end
  
  def generate_clustered_data(n_points_per_cluster: 50, n_features: 50, n_clusters: 3)
    data = []
    labels = []
    
    n_clusters.times do |cluster_id|
      # Keep values smaller and normalized to avoid UMAP issues
      center = Array.new(n_features) { (rand - 0.5) * 0.3 + cluster_id * 0.3 }
      
      n_points_per_cluster.times do
        point = center.map { |c| c + (rand - 0.5) * 0.1 }
        data << point
        labels << cluster_id
      end
    end
    
    # Add some noise points for HDBSCAN testing
    (n_points_per_cluster * 0.2).to_i.times do
      point = Array.new(n_features) { rand * 2 - 1 }  # Random noise
      data << point
      labels << -1  # Mark as noise
    end
    
    [data, labels, "Gaussian Clusters with Noise"]
  end
  
  def generate_swiss_roll(n_points: 150)
    data = []
    labels = []
    
    n_points.times do |i|
      t = 0.5 * Math::PI * (1 + 2 * i.to_f / n_points)
      height = rand
      
      x = t * Math.cos(t) * 0.1
      y = height * 0.1
      z = t * Math.sin(t) * 0.1
      
      point = [x, y, z]
      
      # Add correlated features
      10.times do |j|
        point << x * Math.sin(j) + y * Math.cos(j) + (rand - 0.5) * 0.01
      end
      
      # Add random features
      37.times do
        point << rand * 0.01
      end
      
      data << point
      labels << (t / (3 * Math::PI) * 3).to_i
    end
    
    [data, labels, "Swiss Roll"]
  end
  
  def generate_iris_like_data
    data = []
    labels = []
    
    species_params = [
      { sepal_length: 0.5, sepal_width: 0.34, petal_length: 0.15, petal_width: 0.02 },
      { sepal_length: 0.59, sepal_width: 0.28, petal_length: 0.43, petal_width: 0.13 },
      { sepal_length: 0.65, sepal_width: 0.30, petal_length: 0.55, petal_width: 0.20 }
    ]
    
    species_params.each_with_index do |params, species_id|
      50.times do
        features = [
          params[:sepal_length] + (rand - 0.5) * 0.08,
          params[:sepal_width] + (rand - 0.5) * 0.06,
          params[:petal_length] + (rand - 0.5) * 0.08,
          params[:petal_width] + (rand - 0.5) * 0.04
        ]
        
        # Expand to 50 dimensions
        expanded = features.dup
        
        features.each do |f1|
          features.each do |f2|
            expanded << f1 * f2 * 0.01
          end
        end
        
        features.each_with_index do |f, i|
          expanded << Math.sin(f) * 0.01 * (i + 1)
          expanded << Math.cos(f) * 0.01 * (i + 1)
        end
        
        while expanded.length < 50
          expanded << rand * 0.01
        end
        
        data << expanded[0...50]
        labels << species_id
      end
    end
    
    [data, labels, "Iris-like Dataset"]
  end
  
  def generate_visualization_html(data:, umap_data:, pca_data:, svd_data:, true_labels:, 
                                   clustering_results:, dataset_name:, metrics:, clustering_method:)
    # Prepare plots based on clustering method
    plots_html = ""
    
    if clustering_method == 'both'
      # Show both K-means and HDBSCAN side by side
      plots_html = generate_comparison_plots(
        umap_data, pca_data, svd_data, true_labels,
        clustering_results[:kmeans], clustering_results[:hdbscan]
      )
    elsif clustering_method == 'kmeans'
      plots_html = generate_kmeans_plots(
        umap_data, pca_data, svd_data, true_labels,
        clustering_results[:kmeans]
      )
    elsif clustering_method == 'hdbscan'
      plots_html = generate_hdbscan_plots(
        umap_data, pca_data, svd_data, true_labels,
        clustering_results[:hdbscan]
      )
    end
    
    # Generate metrics HTML
    metrics_html = generate_metrics_html(metrics, clustering_method)
    
    <<~HTML
      <!DOCTYPE html>
      <html>
      <head>
          <title>AnnEmbed Visualization - #{dataset_name}</title>
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <style>
              body {
                  font-family: Arial, sans-serif;
                  margin: 20px;
                  background: #f5f5f5;
              }
              h1 {
                  color: #333;
                  text-align: center;
              }
              .container {
                  display: grid;
                  grid-template-columns: repeat(#{clustering_method == 'both' ? 3 : 2}, 1fr);
                  gap: 20px;
                  max-width: #{clustering_method == 'both' ? 1800 : 1400}px;
                  margin: 0 auto;
              }
              .plot {
                  background: white;
                  border-radius: 8px;
                  padding: 10px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              }
              .stats {
                  background: white;
                  border-radius: 8px;
                  padding: 20px;
                  margin: 20px auto;
                  max-width: #{clustering_method == 'both' ? 1800 : 1400}px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
              }
              table {
                  width: 100%;
                  border-collapse: collapse;
              }
              th, td {
                  padding: 10px;
                  text-align: left;
                  border-bottom: 1px solid #ddd;
              }
              th {
                  background-color: #f8f8f8;
                  font-weight: bold;
              }
              .metrics {
                  display: grid;
                  grid-template-columns: repeat(#{clustering_method == 'both' ? 6 : 4}, 1fr);
                  gap: 20px;
                  margin-top: 20px;
              }
              .metric-card {
                  background: #f8f8f8;
                  padding: 15px;
                  border-radius: 5px;
                  text-align: center;
              }
              .metric-value {
                  font-size: 24px;
                  font-weight: bold;
                  color: #333;
              }
              .metric-label {
                  color: #666;
                  margin-top: 5px;
                  font-size: 12px;
              }
              .noise-point {
                  opacity: 0.3;
              }
          </style>
      </head>
      <body>
          <h1>Dimensionality Reduction & Clustering Analysis</h1>
          <h2 style="text-align: center; color: #666;">Dataset: #{dataset_name} | Method: #{clustering_method.capitalize}</h2>
          
          <div class="stats">
              <h2>Dataset Information</h2>
              <table>
                  <tr>
                      <th>Property</th>
                      <th>Value</th>
                  </tr>
                  <tr>
                      <td>Original Dimensions</td>
                      <td>#{data.first.size}</td>
                  </tr>
                  <tr>
                      <td>Number of Points</td>
                      <td>#{data.size}</td>
                  </tr>
                  <tr>
                      <td>True Number of Clusters</td>
                      <td>#{true_labels.reject { |l| l == -1 }.uniq.size}</td>
                  </tr>
              </table>
              
              #{metrics_html}
          </div>
          
          <div class="container">
              #{plots_html}
          </div>
          
          #{generate_additional_plots(metrics, clustering_method)}
          
      </body>
      </html>
    HTML
  end
  
  def generate_metrics_html(metrics, clustering_method)
    html = '<div class="metrics">'
    
    if clustering_method == 'kmeans' || clustering_method == 'both'
      kmeans_metrics = metrics[:kmeans]
      html += <<~HTML
        <div class="metric-card">
            <div class="metric-value">#{kmeans_metrics[:optimal_k]}</div>
            <div class="metric-label">K-means<br>Optimal K</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">#{kmeans_metrics[:silhouette_umap].round(3)}</div>
            <div class="metric-label">K-means UMAP<br>Silhouette</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">#{kmeans_metrics[:silhouette_pca].round(3)}</div>
            <div class="metric-label">K-means PCA<br>Silhouette</div>
        </div>
      HTML
    end
    
    if clustering_method == 'hdbscan' || clustering_method == 'both'
      hdbscan_metrics = metrics[:hdbscan]
      html += <<~HTML
        <div class="metric-card">
            <div class="metric-value">#{hdbscan_metrics[:n_clusters]}</div>
            <div class="metric-label">HDBSCAN<br>Clusters Found</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">#{(hdbscan_metrics[:noise_ratio] * 100).round(1)}%</div>
            <div class="metric-label">HDBSCAN<br>Noise Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">#{hdbscan_metrics[:silhouette_umap].round(3)}</div>
            <div class="metric-label">HDBSCAN UMAP<br>Silhouette</div>
        </div>
      HTML
    end
    
    html += <<~HTML
        <div class="metric-card">
            <div class="metric-value">#{(metrics[:pca_variance_explained] * 100).round(1)}%</div>
            <div class="metric-label">PCA Variance<br>Explained</div>
        </div>
    HTML
    
    html += '</div>'
    html
  end
  
  def generate_comparison_plots(umap_data, pca_data, svd_data, true_labels, kmeans_results, hdbscan_results)
    plots = []
    
    # Row 1: True labels
    plots << create_plot_div('true-umap', umap_data, true_labels, 'UMAP - True Labels', 'UMAP')
    plots << create_plot_div('true-pca', pca_data, true_labels, 'PCA - True Labels', 'PC')
    plots << create_plot_div('true-svd', svd_data, true_labels, 'SVD - True Labels', 'Component')
    
    # Row 2: K-means
    plots << create_plot_div('kmeans-umap', umap_data, kmeans_results[:labels_umap], 'UMAP - K-means', 'UMAP')
    plots << create_plot_div('kmeans-pca', pca_data, kmeans_results[:labels_pca], 'PCA - K-means', 'PC')
    plots << create_plot_div('kmeans-svd', svd_data, kmeans_results[:labels_svd], 'SVD - K-means', 'Component')
    
    # Row 3: HDBSCAN
    plots << create_plot_div('hdbscan-umap', umap_data, hdbscan_results[:labels_umap], 'UMAP - HDBSCAN', 'UMAP', true)
    plots << create_plot_div('hdbscan-pca', pca_data, hdbscan_results[:labels_pca], 'PCA - HDBSCAN', 'PC', true)
    plots << create_plot_div('hdbscan-svd', svd_data, hdbscan_results[:labels_svd], 'SVD - HDBSCAN', 'Component', true)
    
    plots.join("\n")
  end
  
  def generate_kmeans_plots(umap_data, pca_data, svd_data, true_labels, kmeans_results)
    plots = []
    
    # Row 1: True labels
    plots << create_plot_div('true-umap', umap_data, true_labels, 'UMAP - True Labels', 'UMAP')
    plots << create_plot_div('true-pca', pca_data, true_labels, 'PCA - True Labels', 'PC')
    
    # Row 2: K-means
    plots << create_plot_div('kmeans-umap', umap_data, kmeans_results[:labels_umap], 'UMAP - K-means', 'UMAP')
    plots << create_plot_div('kmeans-pca', pca_data, kmeans_results[:labels_pca], 'PCA - K-means', 'PC')
    
    # Row 3: SVD
    plots << create_plot_div('true-svd', svd_data, true_labels, 'SVD - True Labels', 'Component')
    plots << create_plot_div('kmeans-svd', svd_data, kmeans_results[:labels_svd], 'SVD - K-means', 'Component')
    
    plots.join("\n")
  end
  
  def generate_hdbscan_plots(umap_data, pca_data, svd_data, true_labels, hdbscan_results)
    plots = []
    
    # Row 1: True labels
    plots << create_plot_div('true-umap', umap_data, true_labels, 'UMAP - True Labels', 'UMAP')
    plots << create_plot_div('true-pca', pca_data, true_labels, 'PCA - True Labels', 'PC')
    
    # Row 2: HDBSCAN
    plots << create_plot_div('hdbscan-umap', umap_data, hdbscan_results[:labels_umap], 'UMAP - HDBSCAN', 'UMAP', true)
    plots << create_plot_div('hdbscan-pca', pca_data, hdbscan_results[:labels_pca], 'PCA - HDBSCAN', 'PC', true)
    
    # Row 3: SVD
    plots << create_plot_div('true-svd', svd_data, true_labels, 'SVD - True Labels', 'Component')
    plots << create_plot_div('hdbscan-svd', svd_data, hdbscan_results[:labels_svd], 'SVD - HDBSCAN', 'Component', true)
    
    plots.join("\n")
  end
  
  def create_plot_div(id, data, labels, title, axis_prefix, has_noise = false)
    # Handle noise points specially for HDBSCAN
    colors = if has_noise
      labels.map { |l| l == -1 ? 'gray' : l }
    else
      labels
    end
    
    marker_props = if has_noise
      # Make noise points smaller and semi-transparent
      sizes = labels.map { |l| l == -1 ? 5 : 8 }
      opacities = labels.map { |l| l == -1 ? 0.3 : 0.8 }
      "size: [#{sizes.join(',')}], opacity: [#{opacities.join(',')}],"
    else
      "size: 8,"
    end
    
    <<~HTML
      <div class="plot" id="#{id}"></div>
      <script>
        Plotly.newPlot('#{id}', [{
            x: #{data.map { |p| p[0] }.to_json},
            y: #{data.map { |p| p[1] }.to_json},
            mode: 'markers',
            marker: {
                color: #{colors.to_json},
                #{marker_props}
                colorscale: 'Viridis',
                showscale: false
            },
            type: 'scatter'
        }], {
            title: '#{title}',
            xaxis: { title: '#{axis_prefix} 1' },
            yaxis: { title: '#{axis_prefix} 2' },
            height: 400
        });
      </script>
    HTML
  end
  
  def generate_additional_plots(metrics, clustering_method)
    plots = []
    
    if clustering_method == 'kmeans' || clustering_method == 'both'
      if metrics[:kmeans] && metrics[:kmeans][:elbow_results]
        elbow_data = metrics[:kmeans][:elbow_results]
        plots << <<~HTML
          <div class="stats">
              <h2>K-means Elbow Method Results</h2>
              <div id="elbow-plot" style="height: 400px;"></div>
          </div>
          <script>
              const elbowData = #{elbow_data.to_a.sort.to_h.to_json};
              Plotly.newPlot('elbow-plot', [{
                  x: Object.keys(elbowData),
                  y: Object.values(elbowData),
                  mode: 'lines+markers',
                  marker: { size: 10 },
                  line: { width: 2 }
              }], {
                  title: 'Elbow Method - Optimal K Selection',
                  xaxis: { title: 'Number of Clusters (k)' },
                  yaxis: { title: 'Inertia' },
                  height: 400,
                  annotations: [{
                      x: #{metrics[:kmeans][:optimal_k]},
                      y: elbowData[#{metrics[:kmeans][:optimal_k]}],
                      text: 'Optimal K',
                      showarrow: true,
                      arrowhead: 7,
                      ax: 0,
                      ay: -40
                  }]
              });
          </script>
        HTML
      end
    end
    
    plots.join("\n")
  end
end