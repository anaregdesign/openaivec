# vectorize-openai

Simple wrapper of OpenAI for vectorize requests with single request.

## Installation

```bash
pip install git+https://github.com/anaregdesign/vectorize-openai.git
```

## Uninstall

```bash
pip uninstall openaivec
```

## Basic Usage

```python
import os
from openai import AzureOpenAI
from openaivec import VectorizedOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "<your_api_key>"
api_version = "2024-10-21"
azure_endpoint = "https://<your_resource_name>.openai.azure.com"
deployment_name = "<your_deployment_name>"

client = VectorizedOpenAI(
    client=AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint
    ),
    temperature=0.0,
    top_p=1.0,
    model_name=deployment_name,
    system_message="何科の動物ですか？"
)

client.predict(["パンダ", "うさぎ", "コアラ"])  # => ['クマ科', 'ウサギ科', 'コアラ科']
```


## Usage, process with pandas

```python
import pandas as pd

...

df = pd.DataFrame({"name": ["パンダ", "うさぎ", "コアラ"]})

df.assign(
    kind=lambda df: client.predict(df.name)
)
```

the result is:

| name | kind |
|------|------|
| パンダ | クマ科 |
| うさぎ | ウサギ科 |
| コアラ | コアラ科 |
