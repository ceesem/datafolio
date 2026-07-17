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

## Learning Path

**For Beginners:**
1. Start with [Getting Started](getting-started.md)
2. Then read [Snapshots](snapshots.md) to learn about versioning

**For Specific Use Cases:**
- **Experiment tracking** → [Getting Started](getting-started.md) + [CLI Reference](../reference/cli.md)
- **Reproducible research** → [Snapshots](snapshots.md)
- **Curating results for publication** → `archive()` / `copy(follow_lineage=True)` in the [API Reference](../reference/datafolio-api.md#archiving-items)
- **Team collaboration** → [Getting Started](getting-started.md) (Multi-Instance Access section)
- **Sharing files with non-datafolio users** → `item_path()` / `describe(show_paths=True)` in the [API Reference](../reference/datafolio-api.md#sharing-paths-with-collaborators)
- **Model deployment** → [Working with Models](models.md)
- **Custom sklearn pipelines** → [Working with Models](models.md)
- **Cloud storage / large datasets** → [Polars & References](polars.md) (Working with large tables)

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
