# Getting Started

## Install
```bash
pip install openaivec
```

If you want to use in the `uv` project, you can install it with:
```bash
uv add openaivec
```

Some functions about Apache Spark depend on `pyspark`. We can install its dependencies with extra `spark`:

for pip:
```bash
pip install "openaivec[spark]"
```

for `uv`:
```bash
uv add "openaivec[spark]"
```

## Links
- [https://github.com/anaregdesign/openaivec/](https://github.com/anaregdesign/openaivec/)
- [https://pypi.org/project/openaivec/](https://pypi.org/project/openaivec/)


## Quick Start

Here is a simple example of how to use `openaivec` with `pandas`:

```python
import pandas as pd
from openai import OpenAI
from openaivec import pandas_ext

from typing import List

# Set OpenAI Client (optional: this is default client if environment "OPENAI_API_KEY" is set)
pandas_ext.use(OpenAI())

# Set models for responses and embeddings(optional: these are default models)
pandas_ext.responses_model("gpt-4.1-nano")
pandas_ext.embeddings_model("text-embedding-3-small")


fruits: List[str] = ["apple", "banana", "orange", "grape", "kiwi", "mango", "peach", "pear", "pineapple", "strawberry"]
fruits_df = pd.DataFrame({"name": fruits})
```

`frults_df` is a `pandas` DataFrame with a single column `name` containing the names of fruits. We can mutate the Field `name` with the accessor `ai` to add a new column `color` with the color of each fruit.:

```python
fruits_df.assign(
    color=lambda df: df["name"].ai.responses("What is the color of this fruit?")
)
```

The result is a new DataFrame with the same number of rows as `fruits_df`, but with an additional column `color` containing the color of each fruit. The `ai` accessor uses the OpenAI API to generate the responses for each fruit name in the `name` column.


| name       | color   |
|------------|---------|
| apple      | red     |
| banana     | yellow  |
| orange     | orange  |
| grape      | purple  |
| kiwi       | brown   |
| mango      | orange  |
| peach      | orange  |
| pear       | green   |
| pineapple  | brown   |
| strawberry | red     |


Structured Output is also supported. For example, we will translate the name of each fruit into multiple languages. We can use the `ai` accessor to generate a new column `translation` with the translation of each fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian:

```python
from pydantic import BaseModel

class Translation(BaseModel):
    en: str  # English
    fr: str  # French
    ja: str  # Japanese
    es: str  # Spanish
    de: str  # German
    it: str  # Italian
    pt: str  # Portuguese
    ru: str  # Russian

fruits_df.assign(
    translation=lambda df: df["name"].ai.responses(
        instructions="Translate this fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian.",
        response_format=Translation,
    )
)
```

| name       | translation                                                               |
|------------|----------------------------------------------------------------------------|
| apple      | en='Apple' fr='Pomme' ja='リンゴ' es='Manzana' de...                       |
| banana     | en='Banana' fr='Banane' ja='バナナ' es='Banana' de...                      |
| orange     | en='Orange' fr='Orange' ja='オレンジ' es='Naranja' de...                   |
| grape      | en='Grape' fr='Raisin' ja='ブドウ' es='Uva' de='T...                       |
| kiwi       | en='Kiwi' fr='Kiwi' ja='キウイ' es='Kiwi' de='Kiw...                       |
| mango      | en='Mango' fr='Mangue' ja='マンゴー' es='Mango' de...                      |
| peach      | en='Peach' fr='Pêche' ja='モモ' es='Durazno' de...                         |
| pear       | en='Pear' fr='Poire' ja='梨' es='Pera' de='Birn...                         |
| pineapple  | en='Pineapple' fr='Ananas' ja='パイナップル' es='Piñ...                    |
| strawberry | en='Strawberry' fr='Fraise' ja='イチゴ' es='Fresa...                       |


Structured output can be extracted into separate columns using the `extract` method. For example, we can extract the translations into separate columns for each language:

```python
fruits_df.assign(
    translation=lambda df: df["name"].ai.responses(
        instructions="Translate this fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian.",
        response_format=Translation,
    )
).ai.extract("translation")
```

| name       | translation_en | translation_fr | translation_ja | translation_es | translation_de | translation_it | translation_pt | translation_ru |
|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| apple      | Apple          | Pomme          | リンゴ         | Manzana        | Apfel          | Mela           | Maçã           | Яблоко         |
| banana     | Banana         | Banane         | バナナ         | Banana         | Banane         | Banana         | Banana         | Банан          |
| orange     | Orange         | Orange         | オレンジ       | Naranja        | Orange         | Arancia        | Laranja        | Апельсин       |
| grape      | Grape          | Raisin         | ブドウ         | Uva            | Traube         | Uva            | Uva            | Виноград       |
| kiwi       | Kiwi           | Kiwi           | キウイ         | Kiwi           | Kiwi           | Kiwi           | Kiwi           | Киви           |
| mango      | Mango          | Mangue         | マンゴー       | Mango          | Mango          | Mango          | Manga          | Манго          |
| peach      | Peach          | Pêche          | モモ           | Durazno        | Pfirsich       | Pesca          | Pêssego        | Персик         |
| pear       | Pear           | Poire          | 梨             | Pera           | Birne          | Pera           | Pêra           | Груша          |
| pineapple  | Pineapple      | Ananas         | パイナップル   | Piña           | Ananas         | Ananas         | Abacaxi        | Ананас         |
| strawberry | Strawberry     | Fraise         | イチゴ         | Fresa          | Erdbeere       | Fragola        | Morango        | Клубника       |