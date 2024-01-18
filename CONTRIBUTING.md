# Introduction
Welcome, dear contributor. We are glad you are here. This document will guide you through the process of contributing to this project.

# Project Overview
The package **lakera_chainguard** allows you to secure Large Language Model (LLM) applications and agents built with [LangChain](https://www.langchain.com/) from [prompt injection and jailbreaks](https://platform.lakera.ai/docs/prompt_injection) (and [other risks](https://platform.lakera.ai/docs/api)) with [Lakera Guard](https://www.lakera.ai/).

# How to Contribute
## Guidelines
This project is open to contributions from anyone. We welcome contributions of all kinds. If you want to give feedback, report a bug or request a feature, please open an [issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). If you want to contribute code or documentation, please submit a pull request on the [GitHub repository](https://github.com/lakeraai/lakera_langchain_integration). We will review your contribution and merge it if it is in line with our goals for the package.

## Style Guide
We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. We use [Black](https://github.com/psf/black) to automatically format our code. We perform static type checking with [mypy](http://mypy-lang.org/).

These requirements are all enforced by our pre-commit hooks. Make sure to follow the instructions below to set them up.

# Contribution Process
When contributing, we use the [fork and pull request](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) procedure.
## Forking the Repository
Fork the GitHub repository of the project.

## Project Setup
### Python
The Python environment management is done via [poetry](https://python-poetry.org/). Any Python-related command must therefore be run using `poetry run <command>`. This will ensure that the correct environment is used. If the environment is not set up, the command will take care of installing all required dependencies, so you don't need to worry about that.

### Documentation
We use [MkDocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) to automatically generate a documentation based on the docstrings in the source code. Therefore, it is important to always update the source code's docstrings according to your code changes. To check the updated docs, run `poetry run mkdocs serve` which will serve the documentation on your localhost.

We followed [this guide](https://realpython.com/python-project-documentation-with-mkdocs/) to set up the automatic documentation, so look there for inspiration if you want to contribute to the documentation.

### Pre-Commit Hooks
We use [pre-commit](https://pre-commit.com/) to run a series of checks on the code before it is committed. This ensures that the code is formatted correctly, that the tests pass, and that the code is properly typed. To set up the pre-commit hooks, run `poetry run pre-commit install` in the root of the repository.

## Submitting a Pull Request
When you submit a Pull Request, please make sure to include a detailed description of the changes you made, and to reference any issues that are related to the Pull Request. We will review your Pull Request and merge it if it is in line with our goals for the project.

All submitted code must pass our pre-commit hooks and CI pipelines, and must be properly tested. If you are unsure about how to write tests, please ask for help.

# Conclusion
## Acknowledgments
Thanks to all the people who have contributed to this project, and to the people who have inspired us to create it. In particular, thanks to the team at Lakera AI for their positive feedback and support.

## Contact Information
Any questions or comments can be directed to fv (at) lakera (dot) ai.