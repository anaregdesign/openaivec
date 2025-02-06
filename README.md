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

## Using with Microsoft Fabric

[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric/) is a unified, cloud-based analytics platform that
seamlessly integrates data engineering, warehousing, and business intelligence to simplify the journey from raw data to
actionable insights.

This section provides instructions on how to integrate and use `vectorize-openai` within Microsoft Fabric. Follow these
steps:

1. **Download the WHL File:**
    - Visit the [release page](https://github.com/anaregdesign/vectorize-openai/releases)
    - Download the latest `openaivec-*.*.*-*.whl` file.

2. **Create an Environment in Microsoft Fabric:**
    - In Microsoft Fabric, click on **New item** on your own workspace.
    - Select **Environment** to create a new environment for Apache Spark.
    - Determine the environment name, eg. `openai-environment`.
    - ![image](https://github.com/user-attachments/assets/bd1754ef-2f58-46b4-83ed-b335b64aaa1c)
      *Figure: Creating a new Environment in Microsoft Fabric.*

3. **Upload the WHL File via Custom Library:**
    - Once your environment is set up, go to the **Custom Library** section within that environment.
    - Click on **Upload** and select the downloaded `.whl` file.
    - Save and publish to reflect the changes.
    - ![image](https://github.com/user-attachments/assets/57703f15-1a79-4f0f-8f90-7b64ff845e05)
      *Figure: Uploading the WHL file to the Custom Library.*

4. **Use the Environment from a Notebook:**
    - Open a notebook within Microsoft Fabric.
    - Select the environment you created in the previous steps.
    - ![image](https://github.com/user-attachments/assets/36bc2e26-87c8-4d80-bfc5-bca19300751f)
      *Figure: Using custom environment from a notebook.*
    - In the notebook, import and use `openaivec.spark.UDFBuilder` as you normally would. For example:

      ```python
      from openaivec.spark import UDFBuilder

      udf = UDFBuilder(
          api_key="<your-api-key>",
          api_version="2024-10-21",
          endpoint="https://<your-resource-name>.openai.azure.com",
          model_name="<your-deployment-name"
      )
      ```



Following these steps allows you to successfully integrate and use `vectorize-openai` within Microsoft Fabric.

## Contributing

We welcome contributions to this project! If you would like to contribute, please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.

### Installing Dependencies

To install the necessary dependencies for development, run:

```bash
poetry install --dev
```

### Code Formatting

To reformat the code, use the following command:

```bash
poetry run black ./openaivec
```

### Linting

To check for linting issues, use the following command:

```bash
poetry run flake8 ./openaivec
```

## Contribution

Join our Discord community for developers: https://discord.gg/vbb83Pgn
