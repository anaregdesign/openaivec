from pydantic import BaseModel


class Entity(BaseModel):
    body: str
    entity_class: str


class Relationship(BaseModel):
    tail: Entity
    head: Entity
    property: str


class RelationshipResponse(BaseModel):
    relationships: list[Relationship]


schemaorg_extraction_prompt = (
    """
<Prompt>
    <Instructions>
        Analyze the provided text to extract entities and their relationships in
        accordance with schema.org guidelines. The extraction should follow these
        steps:
        1. Identify all entities in the text.
        2. For each entity, determine its schema.org class. All classes must begin
           with "Thing" (e.g. "Thing.Person").
        3. Identify relationships between the entities. For each relationship,
           ensure that the property used exists in the corresponding schema.org
           definition. For example, for a "Thing.Person" entity, only properties
           defined on the [schema.org/Person](https://schema.org/Person) page may
           be used.
        4. If no entities or relationships can be identified, return an empty
           relationships array.

        IMPORTANT: Do not invent custom entity classes or relationship properties.
        Every "entity_class" and "property" must be strictly verified against
        schema.org definitions.

        Follow these models for your output:
        - Entity:
          {
            "body": "string",
            "entity_class": "string (e.g. Thing.Person)"
          }
        - Relationship:
          {
            "tail": (Entity),
            "head": (Entity),
            "property": "string"
          }
        - Response:
          {
            "relationships": [
              (Relationship), ...
            ]
          }

        Return the result as a single JSON object. Do not include any additional text.
    </Instructions>
    <OutputFormat>
        Output must be in the following JSON structure:
        {
            "relationships": [
                {
                    "tail": {
                        "body": "EntityOne",
                        "entity_class": "Thing.Person"
                    },
                    "head": {
                        "body": "EntityTwo",
                        "entity_class": "Thing.Person"
                    },
                    "property": "relationProperty"
                },
                ...
            ]
        }
    </OutputFormat>
    <Example>
        <Input>
            <![CDATA[
During the conference, Alice knows Bob and is the spouse of Charlie.
            ]]>
        </Input>
        <Output>
            <![CDATA[
{
    "relationships": [
        {
            "tail": {
                "body": "Alice",
                "entity_class": "Thing.Person"
            },
            "head": {
                "body": "Bob",
                "entity_class": "Thing.Person"
            },
            "property": "knows"
        },
        {
            "tail": {
                "body": "Alice",
                "entity_class": "Thing.Person"
            },
            "head": {
                "body": "Charlie",
                "entity_class": "Thing.Person"
            },
            "property": "spouse"
        }
    ]
}
            ]]>
        </Output>
    </Example>
</Prompt>
    """
)
