# Guides

Ordered by how often you will need them. The first two cover most real use.

**[Your first folio](getting-started.md)**
Create a folio, add a mixed bag of objects, read them back by name, understand
what landed on disk. Ten minutes. Start here.

**[Everyday patterns](everyday.md)**
The moves that recur: one path per analysis, lineage, folio metadata, bulk
reads, batched writes, tidying up, handing over a path.

**[Tables: big, external, and lazy](tables.md)**
The only item type where size forces a decision. Parquet and pandas by default;
Polars lazy scans when a table is too big to load; `reference_table()` when the
data is not yours.

**[Models](models.md)**
joblib vs. skops, what each actually protects you from, and why a custom
transformer must live in an importable module.

**[Sharing a folio](sharing.md)**
Cloud paths, read-only opens, multiple readers, sync vs. fork, and what to send
someone who will never install the package.

**[Snapshots](snapshots.md)**
Naming a state you may need to return to — how it works, what it does not
cover, and when not to bother.

**[Reading a folio without DataFolio](format.md)**
The on-disk format as a documented, public surface: the catalog schema, how to
resolve payloads, and the concurrency rules.

**[What DataFolio is not](limits.md)**
Strengths, limits, sharp edges, and what to reach for instead. Worth reading
before you build on it.

**[Migrating to 2.0](migrating-to-2.0.md)**
Method mapping and behavior changes from 1.x.

## If you are in a hurry

- Save and reload one object: [Your first folio](getting-started.md#add-objects)
- A table too big to load: [Tables](tables.md#when-the-table-is-too-big-to-load)
- Link data you do not own: [Tables](tables.md#when-the-table-is-not-yours)
- Give a colleague access: [Sharing](sharing.md#give-collaborators-the-least-dangerous-thing-that-works)
- Freeze the state the paper used: [Snapshots](snapshots.md)
- Look up a method: [API cheat sheet](../reference/datafolio-api.md)
