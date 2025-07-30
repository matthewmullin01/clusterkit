use magnus::{define_module, function, prelude::*, Error, Value};

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let utils_module = parent.define_module("Utils")?;
    
    utils_module.define_singleton_method(
        "estimate_intrinsic_dimension_rust",
        function!(estimate_intrinsic_dimension, 2),
    )?;
    
    utils_module.define_singleton_method(
        "estimate_hubness_rust",
        function!(estimate_hubness, 1),
    )?;
    
    Ok(())
}

fn estimate_intrinsic_dimension(data: Value, k_neighbors: usize) -> Result<f64, Error> {
    // TODO: Implement using annembed
    Err(Error::new(
        magnus::exception::not_imp_error(),
        "Dimension estimation not implemented yet",
    ))
}

fn estimate_hubness(data: Value) -> Result<Value, Error> {
    // TODO: Implement using annembed
    Err(Error::new(
        magnus::exception::not_imp_error(),
        "Hubness estimation not implemented yet",
    ))
}