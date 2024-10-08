# Documentation

This directory contains the documentation source code for LlamaIndex, available at https://docs.llamaindex.ai.

This guide is made for anyone who's interested in running LlamaIndex documentation locally,
making changes to it and making contributions. LlamaIndex is made by the thriving community
behind it, and you're always welcome to make contributions to the project and the
documentation.

## Build Docs

If you haven't already, clone the LlamaIndex Github repo to a local directory:

```
git clone https://github.com/run-llama/llama_index.git && cd llama_index
```

Documentation has its own, dedicated Python virtual environment, and all the tools and scripts are available from the
`docs` directory:

```
cd llama_index/docs
```

From now on, we assume all the commands will be executed from the `docs` directory.

Install all dependencies required for building docs (mainly `mkdocs` and its extension):

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- `poetry install` - this will install all dependencies needed for building docs

To build the docs and browse them locally run:

```
poetry run serve
```

During the build, notebooks are converted to documentation pages, and this takes several minutes. If you're not
working on the "Examples" section of the documentation, you can run the same command with `--skip-notebooks`:

```
poetry run serve --skip-notebooks
```

> [!IMPORTANT]
> Building the documentation takes a while, so make sure you see the following output before opening the browser:
>
> ```
> ...
> INFO    -  Documentation built in 53.32 seconds
> INFO    -  [16:18:17] Watching paths for changes: 'docs'
> INFO    -  [16:18:17] Serving on http://127.0.0.1:8000/en/stable/
> ```

You can now open your browser at http://localhost:8000/ to view the generated docs. The local server will rebuild the
docs and refresh your browser every time you make changes to the docs.

## Configuration

Part of the configuration in `mkdocs.yml` is generated by a script that takes care of keeping the examples in sync as
well as the API reference for all the packages in this repo.

Running the command `poetry run prepare-for-build` from the `docs` folder will update the `mkdocs.yml` with the latest
changes, along with writing new api reference files.

> [!TIP]
> As a contributor, you wouldn't normally need to run this script, feel free to ask for help in the PR.
