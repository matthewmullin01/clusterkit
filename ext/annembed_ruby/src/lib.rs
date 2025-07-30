use magnus::{define_module, function, prelude::*, Error};

mod embedder;
mod svd;
mod utils;

#[magnus::init]
fn init() -> Result<(), Error> {
    let module = define_module("Annembed")?;
    
    // Initialize submodules
    embedder::init(&module)?;
    svd::init(&module)?;
    utils::init(&module)?;
    
    Ok(())
}