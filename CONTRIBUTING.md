# Contributing to Lakera Chainguard

Welcome, dear contributor. We are glad you are here. This document will guide you through the process of contributing to this project.

## Project Overview

The package `lakera_chainguard` allows you to secure Large Language Model (LLM) applications and agents built with [LangChain](https://www.langchain.com/) from [prompt injection and jailbreaks](https://platform.lakera.ai/docs/prompt_injection) (and [other risks](https://platform.lakera.ai/docs/api)) with [Lakera Guard](https://www.lakera.ai/).

## How to Contribute

This project is open to contributions from anyone. We welcome contributions of all kinds. If you want to give feedback, report a bug or request a feature, please open an [issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). If you want to contribute code or documentation, please submit a pull request on the [GitHub repository](https://github.com/lakeraai/chainguard). We will review your contribution and merge it if it is in line with our goals for the package.

### Style Guide
We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. We use [Black](https://github.com/psf/black) to automatically format our code. We perform static type checking with [mypy](http://mypy-lang.org/).

These requirements are all enforced by our pre-commit hooks. Make sure to follow the instructions below to set them up.

### Contribution Process
When contributing, we use the [fork and pull request](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) procedure.

### Python
The Python environment management is done via [poetry](https://python-poetry.org/). Any Python-related command must therefore be run using `poetry run <command>`. This will ensure that the correct environment is used. If the environment is not set up, the command will take care of installing all required dependencies, so you don't need to worry about that.

### Documentation
We use [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) to automatically generate documentation based on the docstrings in the source code. Therefore, it is important to always update the source code's docstrings according to your code changes. To check the updated docs, run `poetry run mkdocs serve` which will serve the documentation on your localhost.

We followed [this guide](https://realpython.com/python-project-documentation-with-mkdocs/) to set up the automatic documentation, so look there for inspiration if you want to contribute to the documentation.

### Pre-Commit Hooks
We use [pre-commit](https://pre-commit.com/) to run a series of checks on the code before it is committed. This ensures that the code is formatted correctly, that the tests pass, and that the code is properly typed. To set up the pre-commit/pre-push hooks, run `poetry run pre-commit install` in the root of the repository.

### Commiting Changes

After you've made your changes and updated any necessary tests, you can commit your changes using the following steps:

1. Create a branch from `main` with the `feature/` prefix
    ```sh
    git checkout -b feature/<your-branch-name>
    ```
2. Make your changes
3. Stage your changes
    ```sh
    git add .
    ```
4. Commit your changes with a commit message that follows the [Conventional Commits](https://www.conventionalcommits.org/) spec
     Here's an example of a valid commit message:

    ```sh
    git commit -m "feat(endpoints): renamed classifiers to endpoints"
    ```

    **Note**: This repo uses `pre-commit` hooks to run [`ruff`](https://github.com/astral-sh/ruff), [`black`](https://github.com/psf/black), and [`isort`](https://github.com/PyCQA/isort) on your code before committing. If any of these checks fail, your commit will be rejected and you will need to stage the changes again and commit them again.
5. Push your `feature/*` branch to your fork
6. Open a [Pull Request](https://github.com/lakeraai/chainguard/pulls) to the `main` branch from your `feature/*` branch

### Submitting a Pull Request

When you submit a Pull Request, please make sure to include a detailed description of the changes you made, and to reference any issues that are related to the Pull Request. We will review your Pull Request and merge it if it is in line with our goals for the project.

All submitted code must pass our pre-commit hooks and CI pipelines, and must be properly tested. If you are unsure about how to write tests, please ask for help.

## Acknowledgments
Thanks to all the people who have contributed to this project, and to the people who have inspired us to create it. In particular, thanks to the team at Lakera AI for their positive feedback and support.

## Contact Information

Open an [Issue](https://github.com/lakeraai/chainguard/issues) or get in touch with `opensource` (at) `lakera.ai`.