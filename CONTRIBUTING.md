# Contributing to vectorize-openai

We welcome contributions to this project! If you would like to contribute, please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.

## Join our Discord community

Join our Discord community for developers: https://discord.gg/vbb83Pgn

## Installing Dependencies

To install the necessary dependencies for development, run:

```bash
poetry install --dev
```

## Running black

To reformat the code, use the following command:

```bash
poetry run black ./openaivec
```

## Linting Guidance

To maintain code quality and consistency, we use `flake8` for linting. Please ensure your code passes the linting checks before submitting a pull request.

### Running flake8

To run `flake8` and check for linting issues, use the following command:

```bash
poetry run flake8 ./openaivec
```
