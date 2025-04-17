from pydantic import BaseModel

__all__ = [
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
        Analyze the provided text to extract entities and their relationships
        according to schema.org guidelines. Follow these steps:
        1. Identify all entities in the text.
        2. For each entity, determine its schema.org class.
           **IMPORTANT: Return the full hierarchy from "Thing" to the class that
           does not have any child classes using dot-separated notation (e.g.
           "Thing.Person"). Use only classes that exist in schema.org. For instance,
           classify a historical landmark as
           "Thing.Place.LandmarksOrHistoricalBuildings" and a broadcast event as
           "Thing.Event.PublicationEvent.BroadcastEvent".**
        3. Identify relationships between entities using only properties defined
           in the relevant schema.org definitions.
        4. If no entities or relationships are found, return an empty relationships
           array.

        Do not invent or use custom classes or properties; all must be verified
        against schema.org.

        Output models:
        - Entity:
          {
            "body": "string",
            "entity_class": "string (e.g. Thing.Person)"
          }
        - Relationship:
          {
            "tail": (Entity),
            "head": (Entity),
            "property": "string (e.g. birthPlace)"
          }
        - Response:
          {
            "relationships": [
              (Relationship), ...
            ]
          }

        Return a single JSON object with no extra text.
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
The Oscars broadcast took place at the Colosseum, and it was organized by Alice.
            ]]>
        </Input>
        <Output>
            <![CDATA[
{
    "relationships": [
        {
            "tail": {
                "body": "The Oscars broadcast",
                "entity_class": "Thing.Event.PublicationEvent.BroadcastEvent"
            },
            "head": {
                "body": "Colosseum",
                "entity_class": "Thing.Place.LandmarksOrHistoricalBuildings"
            },
            "property": "location"
        },
        {
            "tail": {
                "body": "The Oscars broadcast",
                "entity_class": "Thing.Event.PublicationEvent.BroadcastEvent"
            },
            "head": {
                "body": "Alice",
                "entity_class": "Thing.Person"
            },
            "property": "organizer"
        }
    ]
}
            ]]>
        </Output>
    </Example>
</Prompt>
"""
