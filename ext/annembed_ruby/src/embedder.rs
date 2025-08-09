use magnus::{Error, RArray, RHash, Value, TryConvert, Integer, Float, Module, Object};
use magnus::value::ReprValue;
use hnsw_rs::prelude::*;
use annembed::prelude::*;
use std::fs::File;
use std::io::{Write, Read};
use std::cell::RefCell;
use bincode;
use serde::{Serialize, Deserialize};

// Simple struct to serialize UMAP results
#[derive(Serialize, Deserialize)]
struct SavedUMAPModel {
    n_components: usize,
    n_neighbors: usize,
    embeddings: Vec<Vec<f64>>,
    original_data: Vec<Vec<f32>>,
}

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let umap_class = parent.define_class("RustUMAP", magnus::class::object())?;
    
    umap_class.define_singleton_method("new", magnus::function!(RustUMAP::new, 1))?;
    umap_class.define_singleton_method("load_model", magnus::function!(RustUMAP::load_model, 1))?;
    umap_class.define_method("fit_transform", magnus::method!(RustUMAP::fit_transform, 1))?;
    umap_class.define_method("save_model", magnus::method!(RustUMAP::save_model, 1))?;
    umap_class.define_method("transform", magnus::method!(RustUMAP::transform, 1))?;
    
    Ok(())
}

#[magnus::wrap(class = "AnnEmbed::RustUMAP")]
struct RustUMAP {
    n_components: usize,
    n_neighbors: usize,
    #[allow(dead_code)]
    random_seed: Option<u64>,
    // Store the training data and embeddings for transform approximation
    // Use RefCell for interior mutability
    training_data: RefCell<Option<Vec<Vec<f32>>>>,
    training_embeddings: RefCell<Option<Vec<Vec<f64>>>>,
}

impl RustUMAP {
    fn new(options: RHash) -> Result<Self, Error> {
        let n_components = match options.lookup::<_, Value>(magnus::Symbol::new("n_components")) {
            Ok(val) => {
                if val.is_nil() {
                    2
                } else {
                    Integer::try_convert(val)
                        .map(|i| i.to_u32().unwrap_or(2) as usize)
                        .unwrap_or(2)
                }
            }
            Err(_) => 2,
        };
            
        let n_neighbors = match options.lookup::<_, Value>(magnus::Symbol::new("n_neighbors")) {
            Ok(val) => {
                if val.is_nil() {
                    15
                } else {
                    Integer::try_convert(val)
                        .map(|i| i.to_u32().unwrap_or(15) as usize)
                        .unwrap_or(15)
                }
            }
            Err(_) => 15,
        };
            
        let random_seed = match options.lookup::<_, Value>(magnus::Symbol::new("random_seed")) {
            Ok(val) => {
                if val.is_nil() {
                    None
                } else {
                    Integer::try_convert(val)
                        .map(|i| Some(i.to_u64().unwrap_or(42)))
                        .unwrap_or(None)
                }
            }
            Err(_) => None,
        };
        
        Ok(RustUMAP {
            n_components,
            n_neighbors,
            random_seed,
            training_data: RefCell::new(None),
            training_embeddings: RefCell::new(None),
        })
    }
    
    fn fit_transform(&self, data: Value) -> Result<RArray, Error> {
        // Convert Ruby array to Rust Vec<Vec<f64>>
        let ruby_array = RArray::try_convert(data)?;
        let mut rust_data: Vec<Vec<f64>> = Vec::new();
        
        // Get array length
        let array_len = ruby_array.len();
        
        for i in 0..array_len {
            let row = ruby_array.entry::<Value>(i as isize)?;
            let row_array = RArray::try_convert(row).map_err(|_| {
                Error::new(
                    magnus::exception::type_error(),
                    "Expected array of arrays (2D array)",
                )
            })?;
            
            let mut rust_row: Vec<f64> = Vec::new();
            let row_len = row_array.len();
            
            for j in 0..row_len {
                let val = row_array.entry::<Value>(j as isize)?;
                let float_val = if let Ok(f) = Float::try_convert(val) {
                    f.to_f64()
                } else if let Ok(i) = Integer::try_convert(val) {
                    i.to_i64()? as f64
                } else {
                    return Err(Error::new(
                        magnus::exception::type_error(),
                        "All values must be numeric",
                    ));
                };
                rust_row.push(float_val);
            }
            
            if !rust_data.is_empty() && rust_row.len() != rust_data[0].len() {
                return Err(Error::new(
                    magnus::exception::arg_error(),
                    "All rows must have the same length",
                ));
            }
            
            rust_data.push(rust_row);
        }
        
        if rust_data.is_empty() {
            return Err(Error::new(
                magnus::exception::arg_error(),
                "Input data cannot be empty",
            ));
        }
        
        // Convert to Vec<Vec<f32>> for HNSW
        let data_f32: Vec<Vec<f32>> = rust_data.iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();
        
        // Build HNSW graph
        let ef_c = 50;
        let max_nb_connection = 70;
        let nb_points = data_f32.len();
        let nb_layer = 16.min((nb_points as f32).ln().trunc() as usize);
        
        let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_points, nb_layer, ef_c, DistL2 {});
        
        // Insert data into HNSW
        let data_with_id: Vec<(&Vec<f32>, usize)> = data_f32.iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        hnsw.parallel_insert(&data_with_id);
        
        // Create KGraph from HNSW
        let kgraph: annembed::fromhnsw::kgraph::KGraph<f32> = annembed::fromhnsw::kgraph::kgraph_from_hnsw_all(&hnsw, self.n_neighbors)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        // Set up embedding parameters
        let mut embed_params = EmbedderParams::default();
        embed_params.asked_dim = self.n_components;
        embed_params.nb_grad_batch = 15;
        embed_params.scale_rho = 1.;
        embed_params.beta = 1.;
        embed_params.b = 1.;
        embed_params.grad_step = 1.;
        embed_params.nb_sampling_by_edge = 10;
        embed_params.dmap_init = true;
        
        // Create embedder and perform embedding
        let mut embedder = Embedder::new(&kgraph, embed_params);
        
        // TODO: Figure out how to set random seed in annembed
        
        let embed_result = embedder.embed()
            .map_err(|_| Error::new(magnus::exception::runtime_error(), "Embedding failed"))?;
        
        if embed_result == 0 {
            return Err(Error::new(magnus::exception::runtime_error(), "No points were embedded"));
        }
        
        // Get embedded data
        let embedded_array = embedder.get_embedded_reindexed();
        
        // Store results in a simpler format
        let mut embeddings = Vec::new();
        for i in 0..embedded_array.nrows() {
            let mut row = Vec::new();
            for j in 0..embedded_array.ncols() {
                row.push(embedded_array[[i, j]] as f64);
            }
            embeddings.push(row);
        }
        
        // Store the training data and embeddings for future transforms
        *self.training_data.borrow_mut() = Some(data_f32.clone());
        *self.training_embeddings.borrow_mut() = Some(embeddings.clone());
        
        // Convert result back to Ruby array
        let result = RArray::new();
        for embedding in &embeddings {
            let row = RArray::new();
            for &val in embedding {
                row.push(val)?;
            }
            result.push(row)?;
        }
        
        Ok(result)
    }
    
    // Save the full model (training data + embeddings + params) for future transforms
    fn save_model(&self, path: String) -> Result<(), Error> {
        // Check if we have training data
        let training_data = self.training_data.borrow();
        let training_embeddings = self.training_embeddings.borrow();
        
        let training_data_ref = training_data.as_ref()
            .ok_or_else(|| Error::new(magnus::exception::runtime_error(), "No model to save. Run fit_transform first."))?;
        let training_embeddings_ref = training_embeddings.as_ref()
            .ok_or_else(|| Error::new(magnus::exception::runtime_error(), "No embeddings to save."))?;
        
        let saved_model = SavedUMAPModel {
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            embeddings: training_embeddings_ref.clone(),
            original_data: training_data_ref.clone(),
        };
        
        let serialized = bincode::serialize(&saved_model)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let mut file = File::create(&path)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        file.write_all(&serialized)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        Ok(())
    }
    
    // Load a full model for transforming new data
    fn load_model(path: String) -> Result<Self, Error> {
        let mut file = File::open(&path)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let saved_model: SavedUMAPModel = bincode::deserialize(&buffer)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        Ok(RustUMAP {
            n_components: saved_model.n_components,
            n_neighbors: saved_model.n_neighbors,
            random_seed: None,
            training_data: RefCell::new(Some(saved_model.original_data)),
            training_embeddings: RefCell::new(Some(saved_model.embeddings)),
        })
    }
    
    // Transform new data using k-NN approximation with the training data
    fn transform(&self, data: Value) -> Result<RArray, Error> {
        // Get training data
        let training_data = self.training_data.borrow();
        let training_embeddings = self.training_embeddings.borrow();
        
        let training_data_ref = training_data.as_ref()
            .ok_or_else(|| Error::new(magnus::exception::runtime_error(), "No model loaded. Load a model or run fit_transform first."))?;
        let training_embeddings_ref = training_embeddings.as_ref()
            .ok_or_else(|| Error::new(magnus::exception::runtime_error(), "No embeddings available."))?;
        
        // Convert input data to Rust format
        let ruby_array = RArray::try_convert(data)?;
        let mut new_data: Vec<Vec<f32>> = Vec::new();
        
        for i in 0..ruby_array.len() {
            let row = ruby_array.entry::<Value>(i as isize)?;
            let row_array = RArray::try_convert(row)?;
            let mut rust_row: Vec<f32> = Vec::new();
            
            for j in 0..row_array.len() {
                let val = row_array.entry::<Value>(j as isize)?;
                let float_val = if let Ok(f) = Float::try_convert(val) {
                    f.to_f64() as f32
                } else if let Ok(i) = Integer::try_convert(val) {
                    i.to_i64()? as f32
                } else {
                    return Err(Error::new(
                        magnus::exception::type_error(),
                        "All values must be numeric",
                    ));
                };
                rust_row.push(float_val);
            }
            new_data.push(rust_row);
        }
        
        // For each new point, find k nearest neighbors in training data
        // and average their embeddings (weighted by distance)
        let k = self.n_neighbors.min(training_data_ref.len());
        let result = RArray::new();
        
        for new_point in &new_data {
            // Calculate distances to all training points
            let mut distances: Vec<(f32, usize)> = Vec::new();
            for (idx, train_point) in training_data_ref.iter().enumerate() {
                let dist = euclidean_distance(new_point, train_point);
                distances.push((dist, idx));
            }
            
            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k_nearest = &distances[..k];
            
            // Weighted average of k nearest embeddings
            let mut avg_embedding = vec![0.0; self.n_components];
            let mut total_weight = 0.0;
            
            for &(dist, idx) in k_nearest {
                let weight = 1.0 / (dist as f64 + 0.001); // Inverse distance weighting
                total_weight += weight;
                
                for (i, &val) in training_embeddings_ref[idx].iter().enumerate() {
                    avg_embedding[i] += val * weight;
                }
            }
            
            // Normalize
            for val in &mut avg_embedding {
                *val /= total_weight;
            }
            
            // Convert to Ruby array
            let row = RArray::new();
            for val in avg_embedding {
                row.push(val)?;
            }
            result.push(row)?;
        }
        
        Ok(result)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}