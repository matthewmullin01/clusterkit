use magnus::{Error, RArray, RHash, Value, TryConvert, Integer, Float, Module, Object};
use magnus::value::ReprValue;
use hnsw_rs::prelude::*;
use annembed::prelude::*;
use std::fs::File;
use std::io::{Write, Read};
use bincode;
use serde::{Serialize, Deserialize};

// Struct to hold the trained model state
#[derive(Serialize, Deserialize)]
struct UMAPModel {
    n_components: usize,
    n_neighbors: usize,
    embeddings: Vec<Vec<f64>>,  // The trained embeddings
    original_data: Vec<Vec<f32>>,  // Store original data for transform
    graph_data: Vec<u8>,  // Serialized HNSW graph
}

#[magnus::wrap(class = "AnnEmbed::RustUMAP")]
pub struct RustUMAP {
    n_components: usize,
    n_neighbors: usize,
    random_seed: Option<u64>,
    model: Option<UMAPModel>,  // Store the trained model
    hnsw: Option<Hnsw<f32, DistL2>>,  // Keep HNSW for transform
    embedder: Option<Box<Embedder<f32>>>,  // Keep embedder for transform
}

impl RustUMAP {
    pub fn new(options: RHash) -> Result<Self, Error> {
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
            model: None,
            hnsw: None,
            embedder: None,
        })
    }
    
    fn convert_ruby_to_rust(&self, data: Value) -> Result<Vec<Vec<f64>>, Error> {
        let ruby_array = RArray::try_convert(data)?;
        let mut rust_data: Vec<Vec<f64>> = Vec::new();
        
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
        
        Ok(rust_data)
    }
    
    pub fn fit(&mut self, data: Value) -> Result<(), Error> {
        let rust_data = self.convert_ruby_to_rust(data)?;
        
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
        let kgraph: annembed::fromhnsw::kgraph::KGraph<f32> = 
            annembed::fromhnsw::kgraph::kgraph_from_hnsw_all(&hnsw, self.n_neighbors)
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
        
        let embed_result = embedder.embed()
            .map_err(|_| Error::new(magnus::exception::runtime_error(), "Embedding failed"))?;
        
        if embed_result == 0 {
            return Err(Error::new(magnus::exception::runtime_error(), "No points were embedded"));
        }
        
        // Get embedded data
        let embedded_array = embedder.get_embedded_reindexed();
        
        // Convert embeddings to Vec<Vec<f64>>
        let mut embeddings: Vec<Vec<f64>> = Vec::new();
        for i in 0..embedded_array.nrows() {
            let mut row = Vec::new();
            for j in 0..embedded_array.ncols() {
                row.push(embedded_array[[i, j]] as f64);
            }
            embeddings.push(row);
        }
        
        // Store the model
        self.model = Some(UMAPModel {
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            embeddings: embeddings.clone(),
            original_data: data_f32.clone(),
            graph_data: Vec::new(),  // TODO: Serialize HNSW if needed
        });
        
        // Store HNSW and embedder for transform
        self.hnsw = Some(hnsw);
        self.embedder = Some(Box::new(embedder));
        
        Ok(())
    }
    
    pub fn fit_transform(&mut self, data: Value) -> Result<RArray, Error> {
        // First fit the model
        self.fit(data)?;
        
        // Then return the embeddings
        if let Some(ref model) = self.model {
            let result = RArray::new();
            for embedding in &model.embeddings {
                let row = RArray::new();
                for &val in embedding {
                    row.push(val)?;
                }
                result.push(row)?;
            }
            Ok(result)
        } else {
            Err(Error::new(
                magnus::exception::runtime_error(),
                "Model not fitted",
            ))
        }
    }
    
    pub fn transform(&self, data: Value) -> Result<RArray, Error> {
        if self.model.is_none() {
            return Err(Error::new(
                magnus::exception::runtime_error(),
                "Model must be fitted before transform",
            ));
        }
        
        let new_data = self.convert_ruby_to_rust(data)?;
        let new_data_f32: Vec<Vec<f32>> = new_data.iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();
        
        // For now, we'll use a simple approach:
        // Find nearest neighbors in original data and use their embeddings
        // This is a placeholder - a proper implementation would use the embedder
        
        let model = self.model.as_ref().unwrap();
        let result = RArray::new();
        
        for new_point in &new_data_f32 {
            // Find nearest neighbor in original data
            let mut min_dist = f32::MAX;
            let mut nearest_idx = 0;
            
            for (idx, orig_point) in model.original_data.iter().enumerate() {
                let dist = new_point.iter()
                    .zip(orig_point.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = idx;
                }
            }
            
            // Use the embedding of the nearest neighbor
            let row = RArray::new();
            for &val in &model.embeddings[nearest_idx] {
                row.push(val)?;
            }
            result.push(row)?;
        }
        
        Ok(result)
    }
    
    pub fn save(&self, path: String) -> Result<(), Error> {
        if let Some(ref model) = self.model {
            let serialized = bincode::serialize(model)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
            
            let mut file = File::create(&path)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
            
            file.write_all(&serialized)
                .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
            
            Ok(())
        } else {
            Err(Error::new(
                magnus::exception::runtime_error(),
                "Model must be fitted before saving",
            ))
        }
    }
    
    pub fn load(path: String) -> Result<Self, Error> {
        let mut file = File::open(&path)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        let model: UMAPModel = bincode::deserialize(&buffer)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), e.to_string()))?;
        
        Ok(RustUMAP {
            n_components: model.n_components,
            n_neighbors: model.n_neighbors,
            random_seed: None,
            model: Some(model),
            hnsw: None,  // TODO: Rebuild from serialized data if needed
            embedder: None,
        })
    }
    
    pub fn is_fitted(&self) -> bool {
        self.model.is_some()
    }
}

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let umap_class = parent.define_class("RustUMAP", magnus::class::object())?;
    
    umap_class.define_singleton_method("new", magnus::function!(RustUMAP::new, 1))?;
    umap_class.define_singleton_method("load", magnus::function!(RustUMAP::load, 1))?;
    umap_class.define_method("fit", magnus::method!(RustUMAP::fit, 1))?;
    umap_class.define_method("fit_transform", magnus::method!(RustUMAP::fit_transform, 1))?;
    umap_class.define_method("transform", magnus::method!(RustUMAP::transform, 1))?;
    umap_class.define_method("save", magnus::method!(RustUMAP::save, 1))?;
    umap_class.define_method("fitted?", magnus::method!(RustUMAP::is_fitted, 0))?;
    
    Ok(())
}