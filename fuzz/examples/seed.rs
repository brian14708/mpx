//! Write deterministic starting cases for every fuzz target.

use std::{fs, io, path::Path};

use mpx_fuzz::seeds::catalog;

fn main() -> io::Result<()> {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let seeds = catalog();
    for seed in &seeds {
        let directory = corpus.join(seed.target);
        fs::create_dir_all(&directory)?;
        fs::write(directory.join(format!("seed-{}", seed.name)), &seed.bytes)?;
    }
    println!("Wrote {} seeds to {}", seeds.len(), corpus.display());
    Ok(())
}
