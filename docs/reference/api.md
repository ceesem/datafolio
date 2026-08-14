---
title: API documentation
---

# Package API

`datafolio` exports one class and three exceptions. The class has its own
page — see [DataFolio Methods](datafolio-api.md) for every method and
property. Everything else in the package (handlers, storage backends,
readers) is internal and not part of the public API.

## Exceptions

::: datafolio.ConcurrentWriteError
    options:
        show_source: false
        heading_level: 3

::: datafolio.ManifestReadError
    options:
        show_source: false
        heading_level: 3

::: datafolio.UnsupportedManifestVersionError
    options:
        show_source: false
        heading_level: 3
