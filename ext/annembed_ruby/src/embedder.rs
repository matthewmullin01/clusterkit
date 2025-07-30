use magnus::{Error, RArray, RHash, Value, TryConvert, Integer, Float, Module, Object};
use magnus::value::ReprValue;
use hnsw_rs::prelude::*;
use annembed::prelude::*;

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let umap_class = parent.define_class("RustUMAP", magnus::class::object())?;
    
    umap_class.define_singleton_method("new", magnus::function!(RustUMAP::new, 1))?;
    umap_class.define_method("fit_transform", magnus::method!(RustUMAP::fit_transform, 1))?;
    
    Ok(())
}

#[magnus::wrap(class = "Annembed::RustUMAP")]
struct RustUMAP {
    n_components: usize,
    n_neighbors: usize,
    #[allow(dead_code)]
    random_seed: Option<u64>,
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
        
        // Convert result back to Ruby array
        let result = RArray::new();
        for i in 0..embedded_array.nrows() {
            let row = RArray::new();
            for j in 0..embedded_array.ncols() {
                row.push(embedded_array[[i, j]] as f64)?;
            }
            result.push(row)?;
        }
        
        Ok(result)
    }
}