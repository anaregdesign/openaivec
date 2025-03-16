# What is this?
Define a LLM based UDF for Apache Spark with simple code!

```python
# Model for structured output
class Fruit(pydantic.BaseModel):
   name: str
   color: str
   taste: str

# Prompt with example
prompt = """
    return the color and taste of given fruit.

    #example

    ## input
    apple

    ## output
    {{
        "name": "apple",
        "color": "red",
        "taste": "sweet"
    }}
"""
# Simple UDF builder in openaivec
udf = UDFBuilder.of_openai(...)

# Register UDFs with structured output
spark.udf.register("parse_fruit", udf.completion(prompt, response_format=Fruit))

# Use UDFs in Spark SQL
spark.sql("SELECT name, parse_fruit(name) from dummy").show(truncate=False)
```

The following output is produced:
```text
+------+--------------------------------------------------------+
|name  |fruit(name)                                             |
+------+--------------------------------------------------------+
|apple |{"name":"apple","color":"red","taste":"sweet"}          |
|banana|{"name":"banana","color":"yellow","taste":"sweet"}      |
|cherry|{"name":"cherry","color":"red","taste":"sweet and tart"}|
+------+--------------------------------------------------------+
```

# Overview

This package provides a vectorized interface for the OpenAI API, enabling you to process multiple inputs with a single
API call instead of sending requests one by one.
This approach helps reduce latency and simplifies your code.

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
from openai import OpenAI
from openaivec import VectorizedOpenAI


# Initialize the vectorized client with your system message and parameters
client = VectorizedOpenAI(
    client=OpenAI(),
    temperature=0.0,
    top_p=1.0,
    model_name="<your-model-name>",
    system_message="Please answer only with 'xx family' and do not output anything else."
)

result = client.predict(["panda", "rabbit", "koala"])
print(result)  # Expected output: ['bear family', 'rabbit family', 'koala family']
```

See [examples/basic_usage.ipynb](examples/basic_usage.ipynb) for a complete example.

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

Below is an example showing how to create UDFs for Apache Spark using the provided `UDFBuilder`.
This configuration is intended for use with Azure OpenAI or OpenAI.

```python
from openaivec.spark import UDFBuilder

udf = UDFBuilder.of_azureopenai(
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

## Building Prompts

Building prompt is a crucial step in using LLMs.
In particular, providing a few examples in a prompt can significantly improve an LLM’s performance,
a technique known as "few-shot learning." Typically, a few-shot prompt consists of a purpose, cautions,
and examples.

`FewShotPromptBuilder` is a class that helps you build a few-shot learning prompt with simple interface.

### Basic Usage

`FewShotPromptBuilder` requires simply a purpose, cautions, and examples, and `build` method will 
return rendered prompt with XML format.

Here is an example:

```python
from openaivec.prompt import FewShotPromptBuilder

prompt: str = (
    FewShotPromptBuilder()
    .purpose("Return the smallest category that includes the given word")
    .caution("Never use proper nouns as categories")
    .example("Apple", "Fruit")
    .example("Car", "Vehicle")
    .example("Tokyo", "City")
    .example("Keiichi Sogabe", "Musician")
    .example("America", "Country")
    .build()
)
print(prompt)
```

The output will be:

```xml

<Prompt>
    <Purpose>Return the smallest category that includes the given word</Purpose>
    <Cautions>
        <Caution>Never use proper nouns as categories</Caution>
    </Cautions>
    <Examples>
        <Example>
            <Source>Apple</Source>
            <Result>Fruit</Result>
        </Example>
        <Example>
            <Source>Car</Source>
            <Result>Vehicle</Result>
        </Example>
        <Example>
            <Source>Tokyo</Source>
            <Result>City</Result>
        </Example>
        <Example>
            <Source>Keiichi Sogabe</Source>
            <Result>Musician</Result>
        </Example>
        <Example>
            <Source>America</Source>
            <Result>Country</Result>
        </Example>
    </Examples>
</Prompt>
```

### Improve with OpenAI

For most users, it can be challenging to write a prompt entirely free of contradictions, ambiguities, or
redundancies.
`FewShotPromptBuilder` provides an `improve` method to refine your prompt using OpenAI's API.

`improve` method will try to eliminate contradictions, ambiguities, and redundancies in the prompt with OpenAI's API,
and iterate the process up to `max_iter` times.

Here is an example:

```python
from openai import OpenAI
from openaivec.prompt import FewShotPromptBuilder

client = OpenAI(...)
model_name = "<your-model-name>"
improved_prompt: str = (
    FewShotPromptBuilder()
    .purpose("Return the smallest category that includes the given word")
    .caution("Never use proper nouns as categories")
    # Examples which has contradictions, ambiguities, or redundancies
    .example("Apple", "Fruit")
    .example("Apple", "Technology")
    .example("Apple", "Company")
    .example("Apple", "Color")
    .example("Apple", "Animal")
    # improve the prompt with OpenAI's API, max_iter is number of iterations to improve the prompt.
    .improve(client, model_name, max_iter=5)
    .build()
)
print(improved_prompt)
```

Then we will get the improved prompt with extra examples, improved purpose, and cautions:

```xml
<Prompt>
    <Purpose>Classify a given word into its most relevant category by considering its context and potential meanings.
        The input is a word accompanied by context, and the output is the appropriate category based on that context.
        This is useful for disambiguating words with multiple meanings, ensuring accurate understanding and
        categorization.
    </Purpose>
    <Cautions>
        <Caution>Ensure the context of the word is clear to avoid incorrect categorization.</Caution>
        <Caution>Be aware of words with multiple meanings and provide the most relevant category.</Caution>
        <Caution>Consider the possibility of new or uncommon contexts that may not fit traditional categories.</Caution>
    </Cautions>
    <Examples>
        <Example>
            <Source>Apple (as a fruit)</Source>
            <Result>Fruit</Result>
        </Example>
        <Example>
            <Source>Apple (as a tech company)</Source>
            <Result>Technology</Result>
        </Example>
        <Example>
            <Source>Java (as a programming language)</Source>
            <Result>Technology</Result>
        </Example>
        <Example>
            <Source>Java (as an island)</Source>
            <Result>Geography</Result>
        </Example>
        <Example>
            <Source>Mercury (as a planet)</Source>
            <Result>Astronomy</Result>
        </Example>
        <Example>
            <Source>Mercury (as an element)</Source>
            <Result>Chemistry</Result>
        </Example>
        <Example>
            <Source>Bark (as a sound made by a dog)</Source>
            <Result>Animal Behavior</Result>
        </Example>
        <Example>
            <Source>Bark (as the outer covering of a tree)</Source>
            <Result>Botany</Result>
        </Example>
        <Example>
            <Source>Bass (as a type of fish)</Source>
            <Result>Aquatic Life</Result>
        </Example>
        <Example>
            <Source>Bass (as a low-frequency sound)</Source>
            <Result>Music</Result>
        </Example>
    </Examples>
</Prompt>
```

## Using with Microsoft Fabric

[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric/) is a unified, cloud-based analytics platform that
seamlessly integrates data engineering, warehousing, and business intelligence to simplify the journey from raw data to
actionable insights.

This section provides instructions on how to integrate and use `vectorize-openai` within Microsoft Fabric. Follow these
steps:

1. **Create an Environment in Microsoft Fabric:**
    - In Microsoft Fabric, click on **New item** in your workspace.
    - Select **Environment** to create a new environment for Apache Spark.
    - Determine the environment name, eg. `openai-environment`.
    - ![image](https://github.com/user-attachments/assets/bd1754ef-2f58-46b4-83ed-b335b64aaa1c)
      *Figure: Creating a new Environment in Microsoft Fabric.*

2. **Add `openaivec` to the Environment from Public Library**
    - Once your environment is set up, go to the **Custom Library** section within that environment.
    - Click on **Add from PyPI** and search for latest version of `openaivec`.
    - Save and publish to reflect the changes.
    - ![image](https://github.com/user-attachments/assets/7b6320db-d9d6-4b89-a49d-e55b1489d1ae)
      *Figure: Add `openaivec` from PyPI to Public Library*

3. **Use the Environment from a Notebook:**
    - Open a notebook within Microsoft Fabric.
    - Select the environment you created in the previous steps.
    - ![image](https://github.com/user-attachments/assets/2457c078-1691-461b-b66e-accc3989e419)
      *Figure: Using custom environment from a notebook.*
    - In the notebook, import and use `openaivec.spark.UDFBuilder` as you normally would. For example:

      ```python
      from openaivec.spark import UDFBuilder
 
      udf = UDFBuilder(
          api_key="<your-api-key>",
          api_version="2024-10-21",
          endpoint="https://<your-resource-name>.openai.azure.com",
          model_name="<your-deployment-name>"
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
poetry run ruff check . --fix
```

## Community

Join our Discord community for developers: https://discord.gg/vbb83Pgn
