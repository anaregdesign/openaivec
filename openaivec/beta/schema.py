from pydantic import BaseModel

__ALL__ = [
    "RelationshipResponse",
    "schemaorg_extraction_prompt",
]


class Entity(BaseModel):
    body: str
    entity_class: str


class Relationship(BaseModel):
    tail: Entity
    head: Entity
    property: str


class RelationshipResponse(BaseModel):
    relationships: list[Relationship]


schemaorg_extraction_prompt = """
<Prompt>
    <Instructions>
        Analyze the provided text and extract entities and their relationships
        according to schema.org guidelines. Follow these steps:
        1. Identify all entities in the text.
        2. For each entity, determine its schema.org class.
           **IMPORTANT: Return the complete hierarchy from the root "Thing" to
           the most specific class using dot-separated notation (e.g. "Thing.Person").**
        3. Identify relationships between entities, using only properties
           defined in the respective schema.org definitions (e.g. for a
           "Thing.Person" entity, use properties from
           [schema.org/Person](https://schema.org/Person)).
        4. If no entities or relationships are found, return an empty relationships
           array.

        Do not invent custom classes or properties; all must be verified against
        schema.org definitions.

        Use these output models:
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

        Return a single JSON object with no additional text.
    </Instructions>
    <OutputFormat>
        Output must be in this JSON structure:
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
