"""
Copy these cells into a Jupyter notebook and run them one by one
Each cell is separated by # %% markers
"""

# %% [markdown]
# # DataFolio Tutorial - Protein Analysis Example
#
# This notebook demonstrates a typical protein analysis workflow using DataFolio.

# %% Setup and imports
import pandas as pd
import numpy as np
from pathlib import Path
from datafolio import DataFolio
import shutil

# Clean up any existing demos
if Path("demo_bundles").exists():
    shutil.rmtree("demo_bundles")
if Path("temp_external_data").exists():
    shutil.rmtree("temp_external_data")

print("✓ Environment ready")

# %% Create "external" datasets (simulating datalake)
# In real usage, these would be your large datasets in S3/GCS
external_data_dir = Path("temp_external_data")
external_data_dir.mkdir(exist_ok=True)

# Large training dataset
training_data = pd.DataFrame(
    {
        "protein_id": [f"PROT_{i:05d}" for i in range(1000)],
        "sequence_length": np.random.randint(50, 500, 1000),
        "hydrophobicity": np.random.randn(1000),
        "charge": np.random.randn(1000),
        "label": np.random.choice(["membrane", "cytoplasmic", "nuclear"], 1000),
    }
)
training_file = external_data_dir / "training_proteins.parquet"
training_data.to_parquet(training_file, index=False)

print(f"Created training data: {len(training_data)} proteins")
training_data.head()

# %% Create a DataFolio bundle
# By default, use exact bundle names (no random suffix)
# Simply specify the full path to your bundle directory
folio = DataFolio(
    "demo_bundles/protein-analysis",
    metadata={
        "experiment": "protein_localization_v2",
        "date": "2024-01-15",
        "scientist": "Dr. Smith",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
        },
    },
    use_random_suffix=False,  # Default: creates exactly 'demo_bundles/protein-analysis/'
)
# Note: Set use_random_suffix=True for automatic collision avoidance

print(f"Bundle created: {folio._bundle_dir}")
print(f"Metadata: {folio.metadata}")

# %% Reference external data (NOT copied into bundle)
folio.reference_table(
    "training_data",
    path=str(training_file),
    table_format="parquet",
    num_rows=len(training_data),
    description="Large training dataset",
    code="training_data.to_parquet(training_file)",
)

print("✓ Referenced external training data")
print("  (Path stored, data NOT copied to bundle)")

# %% Add small results to bundle (these ARE copied)
metrics_df = pd.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "f1_score"],
        "train": [0.945, 0.932, 0.928, 0.930],
        "validation": [0.912, 0.901, 0.898, 0.899],
    }
)

folio.add_table(
    "performance_metrics",
    metrics_df,
    inputs=["training_data"],
    models=["rf_classifier"],
    code="model.evaluate(X_train, y_train)",
)
print("✓ Added performance metrics to bundle")
metrics_df

# %% Train and add a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Prepare features
X_train = training_data[["sequence_length", "hydrophobicity", "charge"]].values
y_train = training_data["label"].values

# Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Add to bundle (writes immediately!)
folio.add_model(
    "rf_classifier",
    model,
    inputs=["training_data"],
    hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
    code="model = RandomForestClassifier(...).fit(X_train_scaled, y_train)",
)
folio.add_model("scaler", scaler, inputs=["training_data"])

print("✓ Model and scaler added to bundle")

# %% Update metadata (auto-saves!)
folio.metadata["training_samples"] = len(training_data)
folio.metadata["final_accuracy"] = 0.912

print("✓ Metadata updated and auto-saved")
print(f"Metadata now has {len(folio.metadata)} keys")

# %% View bundle contents
contents = folio.list_contents()
print("Bundle contents:")
for key, items in contents.items():
    print(f"  {key}: {items}")

# %% View lineage and timestamps (NEW!)
print("\n" + "=" * 60)
print("Lineage Tracking Features")
print("=" * 60)

# View full bundle description with lineage
print(folio.describe())

# %% Query lineage relationships (NEW!)
print("\nQuerying lineage relationships:")

# What inputs did the metrics use?
inputs = folio.get_inputs("performance_metrics")
print(f"  performance_metrics inputs: {inputs}")

# What depends on the training data?
dependents = folio.get_dependents("training_data")
print(f"  training_data dependents: {dependents}")

# Get full lineage graph
graph = folio.get_lineage_graph()
print(f"\nFull lineage graph:")
for item, inputs in graph.items():
    if inputs:
        print(f"  {item} ← {inputs}")

# %% Create a derived experiment (NEW!)
print("\nCreating a derived experiment with copy():")

# Create a variant experiment (e.g., testing different hyperparameters)
folio_v2 = folio.copy(
    new_path="demo_bundles/protein-analysis-v2",
    metadata_updates={
        "experiment": "protein_localization_v2_tuned",
        "parent_experiment": "protein_localization_v2",
        "changes": "Increased max_depth to 15",
    },
    exclude_items=["performance_metrics"],  # We'll regenerate this
    use_random_suffix=False,  # Keep clean names for derived experiments
)

print(f"✓ Created derived bundle: {folio_v2._bundle_dir}")
print(f"  Metadata: {folio_v2.metadata['experiment']}")
print(f"  Items copied: {list(folio_v2._items.keys())}")

# %% Load the bundle (simulating a new session)
bundle_path = folio._bundle_dir

loaded_folio = DataFolio(path=bundle_path)

print(f"✓ Loaded bundle: {bundle_path}")
print(f"Experiment: {loaded_folio.metadata['experiment']}")
print(f"Accuracy: {loaded_folio.metadata['final_accuracy']}")

# %% Read included table (from bundle)
metrics = loaded_folio.get_table("performance_metrics")
print("Performance metrics (from bundle):")
metrics

# %% Read referenced table (from external file)
training = loaded_folio.get_table("training_data")
print(f"Training data loaded from external file: {training.shape}")
training.head()

# %% Load and use the model
loaded_model = loaded_folio.get_model("rf_classifier")
loaded_scaler = loaded_folio.get_model("scaler")

# Make a prediction
sample = training_data.iloc[0:1][["sequence_length", "hydrophobicity", "charge"]].values
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Actual: {training_data.iloc[0]['label']}")

# %% Check the bundle on disk
print("Bundle directory structure:")
for item in sorted(Path(bundle_path).rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(bundle_path)
        size = item.stat().st_size
        print(f"  {rel_path} ({size} bytes)")

# %% View metadata.json (includes auto-timestamps!)
import json

with open(Path(bundle_path) / "metadata.json") as f:
    metadata_on_disk = json.load(f)
print("Metadata on disk (note created_at/updated_at auto-timestamps):")
print(json.dumps(metadata_on_disk, indent=2))

# %% View items.json (includes lineage tracking!)
with open(Path(bundle_path) / "items.json") as f:
    items_on_disk = json.load(f)
print("\nItems manifest on disk (note lineage fields):")
print(json.dumps(items_on_disk, indent=2))

# %% Cleanup
print("\nTo clean up, run:")
print("  shutil.rmtree('demo_bundles')")
print("  shutil.rmtree('temp_external_data')")
