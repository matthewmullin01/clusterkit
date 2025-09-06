use magnus::{function, prelude::*, Error, Value, RArray, RHash, Integer};
use hdbscan::{Hdbscan, HdbscanHyperParams};
use crate::utils::ruby_array_to_vec_vec_f64;

/// Perform HDBSCAN clustering
/// Returns a hash with labels and basic statistics
pub fn hdbscan_fit(
    data: Value,
    min_samples: usize,
    min_cluster_size: usize,
    metric: String,
) -> Result<RHash, Error> {
    // Convert Ruby array to Vec<Vec<f64>> using shared helper
    let data_vec = ruby_array_to_vec_vec_f64(data)?;
    let n_samples = data_vec.len();
    
    // Note: hdbscan crate doesn't support custom metrics directly
    // We'll use the default Euclidean distance for now
    if metric != "euclidean" && metric != "l2" {
        eprintln!("Warning: Current hdbscan version only supports Euclidean distance. Using Euclidean.");
    }
    
    // Adjust parameters to avoid index out of bounds errors
    // The hdbscan crate has issues when min_samples >= n_samples
    let adjusted_min_samples = min_samples.min(n_samples.saturating_sub(1)).max(1);
    let adjusted_min_cluster_size = min_cluster_size.min(n_samples).max(2);
    
    // Create hyperparameters
    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(adjusted_min_cluster_size)
        .min_samples(adjusted_min_samples)
        .build();
    
    // Create HDBSCAN instance and run clustering
    let clusterer = Hdbscan::new(&data_vec, hyper_params);
    
    // Run the clustering algorithm - cluster() returns Result<Vec<i32>, HdbscanError>
    let labels = clusterer.cluster().map_err(|e| {
        Error::new(
            magnus::exception::runtime_error(),
            format!("HDBSCAN clustering failed: {:?}", e)
        )
    })?;
    
    // Convert results to Ruby types
    let ruby = magnus::Ruby::get().unwrap();
    let result = RHash::new();
    
    // Convert labels (i32 to Ruby Integer, -1 for noise)
    let labels_array = RArray::new();
    for &label in labels.iter() {
        labels_array.push(Integer::from_value(
            ruby.eval(&format!("{}", label)).unwrap()
        ).unwrap())?;
    }
    result.aset("labels", labels_array)?;
    
    // For now, we'll create dummy probabilities and outlier scores
    // since the basic hdbscan crate doesn't provide these
    // In the future, we could calculate these ourselves or use a more advanced implementation
    
    // Create probabilities array (all 1.0 for clustered points, 0.0 for noise)
    let probs_array = RArray::new();
    for &label in labels.iter() {
        let prob = if label == -1 { 0.0 } else { 1.0 };
        probs_array.push(prob)?;
    }
    result.aset("probabilities", probs_array)?;
    
    // Create outlier scores array (0.0 for clustered points, 1.0 for noise)
    let outlier_array = RArray::new();
    for &label in labels.iter() {
        let score = if label == -1 { 1.0 } else { 0.0 };
        outlier_array.push(score)?;
    }
    result.aset("outlier_scores", outlier_array)?;
    
    // Create empty cluster persistence hash for now
    let persistence_hash = RHash::new();
    result.aset("cluster_persistence", persistence_hash)?;
    
    Ok(result)
}

/// Initialize HDBSCAN module functions
pub fn init(clustering_module: &magnus::RModule) -> Result<(), Error> {
    clustering_module.define_singleton_method(
        "hdbscan_rust",
        function!(hdbscan_fit, 4),
    )?;
    
    Ok(())
}