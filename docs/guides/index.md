# Advanced Guides

Welcome to the DataFolio advanced guides! These tutorials cover specialized topics and advanced features.

**New to DataFolio?** Start with the [Getting Started](getting-started.md) tutorial first, then come back here for advanced topics.

## Available Guides

### [Working with Models](models.md)
**Save and load ML models with custom transformers**

Complete guide to working with machine learning models in DataFolio:
- Scikit-learn models (standard and custom)
- Custom transformers with sklearn mixins
- Joblib vs. skops serialization formats
- When and how to use `custom=True` for portability
- PyTorch models overview
- Model metadata and lineage tracking
- Common patterns (A/B testing, hyperparameter tuning)
- Best practices and FAQ

**Who should read this:** Anyone working with sklearn pipelines, custom transformers, or deploying models across environments.

**Time to complete:** 20-25 minutes

---

### [Snapshots](snapshots.md)
**Version control for your experiments**

Deep dive into DataFolio's snapshot system:
- Why use snapshots (with real-world scenarios)
- Creating and loading snapshots
- Copy-on-write versioning (efficient storage)
- Comparing and managing snapshots
- Snapshot workflows (paper submissions, A/B testing, hyperparameter tuning)
- Git integration and credential protection
- CLI tools for snapshot management
- Best practices and troubleshooting

**Who should read this:** Anyone who wants to version experiments, maintain reproducibility, or experiment safely without losing good results.

**Time to complete:** 15-20 minutes

---

### [Caching](caching.md)
**Speed up remote data access**

Local caching for cloud-stored bundles:
- Why and when to use caching
- Enabling and configuring caching
- Cache management (status, clearing, invalidation)
- Performance examples and benchmarks
- Team collaboration with shared caches
- Offline work preparation
- Best practices and troubleshooting
- Interaction with Parquet filtering

**Who should read this:** Anyone working with remote bundles (S3, GCS, etc.) or wanting faster repeated data access.

**Time to complete:** 15-20 minutes

---

### [Parquet Optimization](parquet-optimization.md)
**Work efficiently with large datasets**

Advanced techniques for working with large Parquet files:
- Column selection (column pruning) - Read only what you need
- Row filtering (predicate pushdown) - Filter before loading
- Memory optimization strategies
- Working with datasets larger than memory
- Cloud storage optimization and caching
- Integration with PyArrow and DuckDB
- Real-world performance examples
- Best practices and troubleshooting

**Who should read this:** Anyone working with large datasets (>1GB), cloud storage, or wanting to optimize performance.

**Time to complete:** 20-25 minutes

---

## Learning Path

**For Beginners:**
1. Start with [Getting Started](getting-started.md)
2. Then read [Snapshots](snapshots.md) to learn about versioning

**For Specific Use Cases:**
- **Experiment tracking** → [Getting Started](getting-started.md) + [CLI Reference](../reference/cli.md)
- **Reproducible research** → [Snapshots](snapshots.md)
- **Team collaboration** → [Getting Started](getting-started.md) (Multi-Instance Access section)
- **Model deployment** → [Working with Models](models.md)
- **Custom sklearn pipelines** → [Working with Models](models.md)
- **Cloud storage** → [Caching](caching.md) + [Parquet Optimization](parquet-optimization.md)
- **Large datasets** → [Parquet Optimization](parquet-optimization.md)
- **Fast remote access** → [Caching](caching.md)

---

## Additional Resources

- [DataFolio API Reference](../reference/datafolio-api.md) - Complete method documentation
- [CLI Reference](../reference/cli.md) - Command-line tools
- [Complete API](../reference/api.md) - Full API documentation
- [About](../index.md) - Overview and quick examples

## Need Help?

- Check the [Getting Started FAQ](getting-started.md#common-questions)
- See the [Snapshots FAQ](snapshots.md#faq)
- Report issues on [GitHub](https://github.com/ceesem/datafolio/issues)
