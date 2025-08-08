use magnus::{define_module, Error};

mod embedder;
mod svd;
mod utils;

#[cfg(test)]
mod tests;

#[magnus::init]
fn init() -> Result<(), Error> {
    let module = define_module("AnnEmbed")?;
    
    // Initialize submodules
    embedder::init(&module)?;
    svd::init(&module)?;
    utils::init(&module)?;
    
    Ok(())
}