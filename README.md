# Overview

This package provides a vectorized interface for the OpenAI API, enabling you to process multiple inputs with a single
API call instead of sending requests one by one.  
This approach reduces latency and simplifies your code.  
Additionally, it integrates effortlessly with Pandas DataFrames and Apache Spark UDFs, making it easy to incorporate
into your data processing pipelines.

## Features

- Vectorized API requests for processing multiple inputs at once.
- Seamless integration with Pandas DataFrames.
- A UDF builder for Apache Spark.
- Compatibility with multiple OpenAI clients, including Azure OpenAI.

## Requirements

- Python 3.10 or higher

## Installation

Install the package with:

```bash
pip install openaivec
```

If you want to uninstall the package, you can do so with:

```bash
pip uninstall openaivec
```

## Basic Usage

```python
import os
from openai import AzureOpenAI
from openaivec import VectorizedOpenAI

# Set environment variables and configurations
os.environ["AZURE_OPENAI_API_KEY"] = "<your_api_key>"
api_version = "2024-10-21"
azure_endpoint = "https://<your_resource_name>.openai.azure.com"
deployment_name = "<your_deployment_name>"

# Initialize the vectorized client with your system message and parameters
client = VectorizedOpenAI(
    client=AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint
    ),
    temperature=0.0,
    top_p=1.0,
    model_name=deployment_name,
    system_message="Please answer simply with a simple “xx family” and do not output anything else."
)

result = client.predict(["panda", "rabbit", "koala"])
print(result)  # Expected output: ['bear family', 'rabbit family', 'koala family']
```

## Using with Pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({"name": ["panda", "rabbit", "koala"]})

df.assign(
    kind=lambda df: client.predict(df.name)
)
```

Example output:

| name   | kind          |
|--------|---------------|
| panda  | bear family   |
| rabbit | rabbit family |
| koala  | koala family  |

## Using with Apache Spark UDF

Below is an example of creating UDFs for Apache Spark using the provided `UDFBuilder`.  
This configuration is intended for Azure OpenAI.

```python
from openaivec.spark import UDFBuilder

udf = UDFBuilder(
    api_key="<your-api-key>",
    api_version="2024-10-21",
    endpoint="https://<your_resource_name>.openai.azure.com",
    model_name="<your_deployment_name>"
)

# Register UDFs (e.g., to extract flavor or product type from product names)
spark.udf.register("parse_taste", udf.completion("""
- Extract flavor-related information from the product name. Return only the concise flavor name with no extra text.
- Minimize unnecessary adjectives related to the flavor.
    - Example:
        - Hokkaido Milk → Milk
        - Uji Matcha → Matcha
"""))

# Register UDFs (e.g., to extract product type from product names)
spark.udf.register("parse_product", udf.completion("""
- Extract the type of food from the product name. Return only the food category with no extra text.
- Example output:
    - Smoothie
    - Milk Tea
    - Protein Bar
"""))
```

You can then use the UDFs in your Spark SQL queries as follows:

```sql
SELECT id,
       product_name,
       parse_taste(product_name)   AS taste,
       parse_product(product_name) AS product
FROM product_names;
```

Example Output:

| id            | product_name                         | taste     | product     |
|---------------|--------------------------------------|-----------|-------------|
| 4414732714624 | Cafe Mocha Smoothie (Trial Size)     | Mocha     | Smoothie    |
| 4200162318339 | Dark Chocolate Tea (New Product)     | Chocolate | Tea         |
| 4920122084098 | Cafe Mocha Protein Bar (Trial Size)  | Mocha     | Protein Bar |
| 4468864478874 | Dark Chocolate Smoothie (On Sale)    | Chocolate | Smoothie    |
| 4036242144725 | Uji Matcha Tea (New Product)         | Matcha    | Tea         |
| 4847798245741 | Hokkaido Milk Tea (Trial Size)       | Milk      | Milk Tea    |
| 4449574211957 | Dark Chocolate Smoothie (Trial Size) | Chocolate | Smoothie    |
| 4127044426148 | Fruit Mix Tea (Trial Size)           | Fruit     | Tea         |
| ...           | ...                                  | ...       | ...         |




