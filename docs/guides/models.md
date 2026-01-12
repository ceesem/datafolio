# Working with Models

This guide covers everything you need to know about storing and loading machine learning models in DataFolio, including scikit-learn models and custom transformers.

## Quick Start

```python
from datafolio import DataFolio
from sklearn.ensemble import RandomForestClassifier

# Train a model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save it
folio = DataFolio('experiments/my_experiment')
folio.add_sklearn('classifier', clf,
    description='Random forest baseline',
    inputs=['training_data'])

# Load it later
loaded_clf = folio.get_sklearn('classifier')
predictions = loaded_clf.predict(X_test)
```

## Scikit-learn Models

### Standard Models

DataFolio automatically handles all standard scikit-learn models and many popular ML libraries:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# All of these work automatically
folio.add_sklearn('rf', RandomForestClassifier())
folio.add_sklearn('gbr', GradientBoostingRegressor())
folio.add_sklearn('lr', LogisticRegression())
folio.add_sklearn('svm', SVC())
folio.add_sklearn('xgb', xgb.XGBClassifier())
folio.add_sklearn('lgb', lgb.LGBMRegressor())
```

### Custom Transformers with Skops

When you create **custom transformers** for sklearn pipelines, you need to use the `custom=True` flag to enable [skops](https://skops.readthedocs.io/) serialization. This makes your pipelines portable across different environments.

#### Why Use Skops?

**Use skops (`custom=True`) when:**
- Your pipeline contains custom transformers (not from sklearn/standard libraries)
- You need to deploy models to environments without access to your class definitions
- You want more secure model serialization for production
- You're sharing models with collaborators who may not have your codebase

**Use joblib (default) when:**
- All components are from standard libraries (sklearn, XGBoost, LightGBM, etc.)
- You're working within a single environment/codebase
- You prioritize speed over portability

#### Creating Custom Transformers

**The key requirement:** Custom transformers MUST inherit from sklearn's base classes.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# ✅ CORRECT - Inherits from sklearn base classes
class PercentileClipper(BaseEstimator, TransformerMixin):
    """Custom transformer that clips values to percentile bounds."""

    def __init__(self, lower=1, upper=99):
        """Initialize with percentile bounds.

        Args:
            lower: Lower percentile bound (default: 1)
            upper: Upper percentile bound (default: 99)
        """
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        """Fit by computing percentile bounds.

        Args:
            X: Training data
            y: Target values (ignored)

        Returns:
            self
        """
        self.lower_bound_ = np.percentile(X, self.lower, axis=0)
        self.upper_bound_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        """Transform by clipping to percentile bounds.

        Args:
            X: Data to transform

        Returns:
            Clipped data
        """
        return np.clip(X, self.lower_bound_, self.upper_bound_)


# ❌ WRONG - Plain class without sklearn mixins
class BadTransformer:
    """This won't work with skops!"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
```

#### Why Inherit from BaseEstimator and TransformerMixin?

These mixins provide essential sklearn functionality:

**`BaseEstimator` provides:**
- `get_params()` - Required by sklearn for introspection
- `set_params()` - Required by sklearn for hyperparameter tuning
- Ensures your transformer works with GridSearchCV, RandomizedSearchCV, etc.

**`TransformerMixin` provides:**
- `fit_transform()` - Convenience method that calls fit() then transform()
- Ensures your transformer works seamlessly in pipelines

**Required for skops:**
- Skops needs these methods to properly serialize and deserialize your custom classes
- Without them, skops cannot reconstruct your transformer when loading

#### Using Custom Transformers in Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline with custom transformer
pipeline = Pipeline([
    ('clipper', PercentileClipper(lower=5, upper=95)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Fit pipeline
X_train = np.random.randn(100, 5)
y_train = np.random.randint(0, 2, 100)
pipeline.fit(X_train, y_train)

# Save with skops format
folio.add_sklearn('custom_pipeline', pipeline,
    custom=True,  # ← Important! Enables skops
    description='Pipeline with custom percentile clipper',
    inputs=['training_data'])

# Load in a different environment
# No need for PercentileClipper class definition!
folio2 = DataFolio('experiments/my_experiment')
loaded_pipeline = folio2.get_sklearn('custom_pipeline')
predictions = loaded_pipeline.predict(X_test)
```

#### Best Practices for Custom Transformers

**1. Always inherit from sklearn base classes**

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    # Your implementation
    pass
```

**2. Store all parameters in `__init__`**

```python
def __init__(self, threshold=0.5, method='mean'):
    # Store ALL parameters - required for get_params()
    self.threshold = threshold
    self.method = method
```

**3. Store fitted parameters with trailing underscore**

```python
def fit(self, X, y=None):
    # Fitted parameters end with underscore (sklearn convention)
    self.mean_ = np.mean(X)
    self.std_ = np.std(X)
    return self
```

**4. Always return `self` from `fit()`**

```python
def fit(self, X, y=None):
    # Do fitting...
    return self  # ← Required for sklearn API
```

**5. Make `transform()` stateless**

```python
def transform(self, X):
    # Only use fitted parameters (those ending with _)
    # Don't modify instance state here
    return (X - self.mean_) / self.std_
```

#### Serialization Format Comparison

| Feature | Joblib (default) | Skops (`custom=True`) |
|---------|------------------|----------------------|
| **Speed** | Faster | Slightly slower |
| **Portability** | Requires class definitions | Self-contained |
| **Use case** | Standard libraries only | Custom transformers |
| **Security** | Less secure | More secure |
| **Deployment** | Need codebase | Standalone |

```python
# Joblib format (default)
folio.add_sklearn('model', pipeline)
# → Saves as .joblib
# → Fast but requires class definitions

# Skops format (portable)
folio.add_sklearn('model', pipeline, custom=True)
# → Saves as .skops
# → Self-contained, works without class definitions
```

### Complete Example: Custom Transformer Pipeline

```python
from datafolio import DataFolio
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define custom transformer
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips outliers based on IQR method."""

    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        self.lower_bound_ = q1 - self.iqr_multiplier * iqr
        self.upper_bound_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)

# Create and train pipeline
pipeline = Pipeline([
    ('outlier_clipper', OutlierClipper(iqr_multiplier=1.5)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Generate training data
X_train = np.random.randn(200, 10)
y_train = np.random.randint(0, 2, 200)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Save with skops
folio = DataFolio('experiments/outlier_detection')
folio.add_sklearn('pipeline', pipeline,
    custom=True,  # Enable skops for custom transformer
    description='Logistic regression with IQR-based outlier clipping',
    inputs=['training_data'],
    hyperparameters={
        'iqr_multiplier': 1.5,
        'random_state': 42
    })

# Later: Load and use (even without OutlierClipper class!)
folio2 = DataFolio('experiments/outlier_detection')
loaded_pipeline = folio2.get_sklearn('pipeline')

# Make predictions
X_test = np.random.randn(50, 10)
predictions = loaded_pipeline.predict(X_test)
probabilities = loaded_pipeline.predict_proba(X_test)

print(f"Predictions: {predictions[:5]}")
print(f"Probabilities: {probabilities[:5]}")
```

## Model Metadata

Add rich metadata to track model provenance:

```python
folio.add_sklearn('classifier', model,
    description='Random forest with balanced class weights',
    inputs=['processed_features', 'labels'],
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'random_state': 42
    },
    code='''
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    ''')
```

## Loading Models

All models can be loaded with type-specific or generic methods:

```python
# Type-specific method
clf = folio.get_sklearn('classifier')

# Generic method (delegates to get_sklearn)
clf = folio.get_model('classifier')

# Data accessor (autocomplete-friendly)
clf = folio.data.classifier.content
```

## Common Patterns

### A/B Testing Models

```python
# Train baseline
baseline = RandomForestClassifier(n_estimators=50)
baseline.fit(X_train, y_train)

# Train variant
variant = RandomForestClassifier(n_estimators=200)
variant.fit(X_train, y_train)

# Save both
folio.add_sklearn('baseline', baseline)
folio.add_sklearn('variant', variant)

# Compare
baseline_score = folio.data.baseline.content.score(X_test, y_test)
variant_score = folio.data.variant.content.score(X_test, y_test)

# Deploy winner
if variant_score > baseline_score:
    production_model = folio.get_sklearn('variant')
else:
    production_model = folio.get_sklearn('baseline')
```

### Pipeline Versioning

```python
# Version 1: Simple pipeline
v1_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
v1_pipeline.fit(X_train, y_train)
folio.add_sklearn('pipeline_v1', v1_pipeline)

# Version 2: Added custom preprocessing
v2_pipeline = Pipeline([
    ('clipper', PercentileClipper()),  # Custom transformer
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
v2_pipeline.fit(X_train, y_train)
folio.add_sklearn('pipeline_v2', v2_pipeline, custom=True)  # Need skops!

# Compare versions
v1_score = folio.data.pipeline_v1.content.score(X_test, y_test)
v2_score = folio.data.pipeline_v2.content.score(X_test, y_test)
```

### Hyperparameter Tuning Archive

```python
from sklearn.model_selection import ParameterGrid

# Define grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

# Try all combinations
for params in ParameterGrid(param_grid):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Save each model
    name = f"rf_n{params['n_estimators']}_d{params['max_depth']}"
    folio.add_sklearn(name, model,
        hyperparameters=params,
        description=f"RF with {params['n_estimators']} trees, depth {params['max_depth']}")

    # Track score in metadata
    folio._items[name]['test_score'] = score

# Find best model
best_name = max(folio.models,
    key=lambda name: folio._items[name].get('test_score', 0))
best_model = folio.get_sklearn(best_name)
print(f"Best model: {best_name}")
print(f"Score: {folio._items[best_name]['test_score']}")
```

## FAQ

**Q: When should I use `custom=True`?**

A: Use it when your pipeline contains custom transformers (classes you wrote). Standard sklearn/XGBoost/LightGBM models don't need it.

**Q: Can I mix joblib and skops models in the same bundle?**

A: Yes! DataFolio automatically detects the format when loading. You can have some models saved with joblib and others with skops.

**Q: Do I need to install skops?**

A: Only if you use `custom=True`. For standard models (joblib format), skops is not required.

**Q: Can I convert a joblib model to skops?**

A: Yes, just load and re-save with `custom=True`:
```python
model = folio.get_sklearn('old_model')
folio.add_sklearn('new_model', model, custom=True)
```

**Q: What if I don't inherit from BaseEstimator/TransformerMixin?**

A: Skops serialization will fail. Always inherit from these classes for custom transformers.

**Q: How do I know which format a model uses?**

A: Check the metadata:
```python
print(folio._items['model_name']['serialization_format'])  # 'joblib' or 'skops'
print(folio._items['model_name']['filename'])  # ends in .joblib or .skops
```

## Next Steps

- [Getting Started Guide](getting-started.md) - Complete tutorial
- [API Reference](../reference/datafolio-api.md) - Method documentation
- [Snapshots](snapshots.md) - Version control for models
- [GitHub Examples](https://github.com/ceesem/datafolio/tree/main/examples) - More examples
