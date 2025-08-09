# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Clean, scikit-learn-like API for UMAP
  - `fit(data)` - Train the model
  - `transform(data)` - Transform new data
  - `fit_transform(data)` - Train and transform in one step
  - `fitted?` - Check if model is trained
  - `save(path)` - Save trained model
  - `load(path)` - Load trained model
- Model persistence with save/load functionality
- Data export/import utilities for caching results
- Comprehensive test suite for UMAP interface
- Detailed README with practical examples

### Changed
- Complete API redesign to follow ML library conventions
- Removed confusing `save_embeddings`/`load_embeddings` methods
- Separated model operations from data caching concerns

### Fixed
- Intermittent test failures with boundary assertions
- Data normalization issues with extreme values

## [0.1.0] - TBD

### Added
- Initial release with basic embedding functionality