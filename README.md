# ranking-metrics

[![PyPI version](https://img.shields.io/pypi/v/trysac-ranking-metrics)](https://pypi.org/project/trysac-ranking-metrics/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-ranking-metrics)](LICENSE)
[![CI](https://github.com/Trysac/u229-ranking-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u229-ranking-metrics/actions/workflows/ci.yml)

Rank evaluation over TREC qrels and run files: nDCG@k, MAP, MRR and P@k with trec_eval conventions, plus a paired t-test and an exact or seeded randomization test

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-ranking-metrics
# or
uv add trysac-ranking-metrics
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import ranking_metrics

# normalize() trims surrounding whitespace and lower-cases the text.
print(ranking_metrics.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    ranking_metrics.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
