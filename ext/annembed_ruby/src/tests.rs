#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_compilation() {
        // Simple test to ensure Rust code compiles
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn test_vector_conversion() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].len(), 2);
    }
}