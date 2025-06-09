# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies
```bash
# Install all dependencies including development and optional extras
uv sync --all-extras --dev
```

### Testing
```bash
# Run all tests (requires Java 17+ for Spark tests)
JAVA_HOME=$(/usr/libexec/java_home -v 17) PATH=$(/usr/libexec/java_home -v 17)/bin:$PATH uv run pytest tests/ -v

# Run tests excluding Spark (if Java 17+ not available)
uv run pytest tests/ --ignore=tests/test_spark.py -v

# Run specific test file
uv run pytest tests/test_responses.py -v

# Run single test
uv run pytest tests/test_responses.py::TestVectorizedResponsesOpenAI::test_predict_str -v
```

### Code Quality
```bash
# Run linter and fix issues
uv run ruff check . --fix

# Check linting without fixing
uv run ruff check .
```

## Architecture Overview

**openaivec** is a vectorized OpenAI API client optimized for batch processing in data science workflows. The library provides three main integration patterns:

### Core Components

1. **Batch Processing Layer** (`responses.py`, `embeddings.py`)
   - `BatchResponses` / `AsyncBatchResponses`: Vectorized text generation with JSON mode support
   - `BatchEmbeddings` / `AsyncBatchEmbeddings`: Efficient embedding generation with deduplication
   - Both support rate limiting, exponential backoff, and concurrent processing

2. **Pandas Integration** (`pandas_ext.py`)
   - `.ai` accessor: Synchronous operations on Series/DataFrames
   - `.aio` accessor: Asynchronous operations for improved performance
   - Methods: `responses()`, `embeddings()`, `count_tokens()`, `extract()`, `similarity()`

3. **Spark Integration** (`spark.py`)
   - `ResponsesUDFBuilder` / `EmbeddingsUDFBuilder`: Factory classes for async Spark UDFs
   - Support for both OpenAI and Azure OpenAI via `.of_openai()` / `.of_azure_openai()` class methods
   - Structured output support using Pydantic models

### Key Design Patterns

- **Deduplication**: All batch operations automatically deduplicate inputs to minimize API costs
- **Async Semaphores**: Concurrency control to respect API rate limits
- **Factory Methods**: Consistent instantiation patterns across sync/async and OpenAI/Azure variants
- **Type Safety**: Extensive use of generics and Pydantic for structured outputs

### Dependencies and Environment

- **Python**: 3.10+ required
- **Java**: 17+ required for Spark functionality (PySpark 3.5.5+)
- **OpenAI API**: Compatible with both OpenAI and Azure OpenAI endpoints
- **Core Dependencies**: `openai`, `pandas`, `tiktoken`
- **Optional Dependencies**: `pyspark` for Spark integration

### Testing Strategy

The test suite uses mock OpenAI clients to avoid API calls during testing. Spark tests require proper Java 17+ environment setup. Tests cover:
- Sync/async batch processing
- Pandas accessor functionality  
- Spark UDF generation and execution
- Error handling and edge cases
- Serialization and type conversion

### Client Configuration

The library supports multiple client setup patterns:
- Environment variables (`OPENAI_API_KEY`, `AZURE_OPENAI_*`)
- Direct client injection via `pandas_ext.use()` / `pandas_ext.use_async()`
- Factory methods with explicit credentials for Spark UDFs