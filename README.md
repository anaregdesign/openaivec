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
    system_message="Please answer simply with a simple “xx family” and do not output anything else."
)

client.predict(["panda", "rabit", "koala"])  # => ['bear family', 'rabbit family', 'koala family']
```


## Usage, process with pandas

```python
import pandas as pd

...

df = pd.DataFrame({"name": ["panda", "rabbit", "koala"]})

df.assign(
    kind=lambda df: client.predict(df.name)
)
```

the result is:

| name   | kind          |
|--------|---------------|
| panda  | bear family   |
| rabbit | rabbit family |
| koala  | koala family  |

## Using Azure OpenAI with Apache Spark UDF

Here's simple example of parsing product names using OpenAI with Apache Spark UDF.

You can use the `openaivec` package to create a UDF function to use with Apache Spark.
At first, you need to create a `UDFConfig` object with the configuration of your OpenAI deployment.

```python
from openaivec import completion_udf, UDFConfig

conf = UDFConfig(
    api_key="<your-api-key>",
    api_version="2024-10-21",
    endpoint="https://<your-resource-name>.openai.azure.com",
    model_name="<your-deployment-name"
)

```

here you can use the `openai_udf` function to create a UDF function to use with Apache Spark.

```python
spark.udf.register("parse_taste", openai_udf(conf, """
- Extract flavor-related information included in the product name. Only output the flavor name concisely, and nothing else.  
- Minimize unnecessary adjectives regarding the flavor as much as possible.  
    - Example:  
        - Hokkaido Milk → Milk  
        - Uji Matcha → Matcha  

"""))

spark.udf.register("parse_product", openai_udf(conf, """
- Extract the type of food included in the product name. Only output the food category and nothing else.  
- Example output:  
    - Smoothie  
    - Milk Tea  
    - Protein Bar  
"""))
```

and then you can use the UDF function in your queries.

```sparksql
select id,
       product_name,
       parse_taste(product_name)   as taste,
       parse_product(product_name) as product
from product_names
```

Output:

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




