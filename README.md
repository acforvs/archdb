# archdb

I wanted a way to compare recent LLMs without digging through a different paper or config for each
one. archdb puts their architectures and pretraining recipes in one table, with links back to the
sources.

[Browse archdb](https://archdb.vladsavinov.com)

## Terminal

Install with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/acforvs/archdb
archdb
```

Browse models and compare their architecture and training fields. Search fields with `/`, find
models with `m`, and press `?` for the keys. The explorer works offline on macOS and Linux.
Update the tool and its bundled data with `uv tool upgrade archdb`.

There are also commands for filtering models and exporting results:

```sh
archdb models --where 'moe.sparsity >= 32' --sort params.active --desc
archdb show deepseek-v3 --group moe
```

See [Terminal usage](docs/cli.md) for filtering, source lookup, KV settings and JSON/CSV output.

## Use the data

The data is also available without the website:

- [models.csv](exports/models.csv) — one row per model.
- [facts.csv](exports/facts.csv) — individual values with their sources and quotes.
- [archdb.json](exports/archdb.json) — the complete dataset.

See [Using the data](docs/data.md) for field definitions, other exports, and calculation details.

## Run locally

Requires Python 3.11+, Node.js 24+, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run archdb build
npm --prefix web ci
npm --prefix web run build
uv run archdb serve
```

Open [localhost:8000](http://localhost:8000). See [Development](docs/development.md) for checks and
deployment commands.

## Contributing

To add a model or fix a value, see [CONTRIBUTING.md](CONTRIBUTING.md). You can also
[open an issue](https://github.com/acforvs/archdb/issues).

## License

Code: [MIT](LICENSE). Data: [CC0 1.0](LICENSE-DATA). Quoted material keeps its original license.
