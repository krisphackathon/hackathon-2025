from pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Workflow,
    StopEvent,
    StartEvent,
    Context,
    step,
    Event,
)

class QueryPlanItem(Event):
    """A single step in an execution plan for a RAG system."""

    name: str = Field(description="The name of the tool to use.")
    query: str = Field(
        description="A natural language search query for a RAG system."
    )


class QueryPlan(BaseModel):
    """A plan for a RAG system. After running the plan, we should have either enough information to answer the user's original query, or enough information to form a new query plan."""

    items: list[QueryPlanItem] = Field(
        description="A list of the QueryPlanItem objects in the plan."
    )


class QueryPlanItemResult(Event):
    """The result of a query plan item"""

    query: str
    result: str


class ExecutedPlanEvent(Event):
    """The result of a query plan"""

    result: str