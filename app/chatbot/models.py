from pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Workflow,
    StopEvent,
    StartEvent,
    Context,
    step,
    Event,
)
from typing import Literal

class QueryPlanItem(Event):
    """A single step in an execution plan for a RAG system."""

    name: str = Field(description="The name of the tool to use.")
    query: str = Field(
        description="A natural language search query for a RAG system."
    )


class QueryPlan(BaseModel):
    """A plan for a RAG system. After running the plan, we should have either enough information to answer the user's original query, or enough information to form a new query plan."""
    reasoning: str = Field(
        description="Your step by step reasoning process on query formulation for semantic search."
    )
    items: list[QueryPlanItem] = Field(
        description="A list of the QueryPlanItem objects in the plan."
    )

class QueryAnswer(BaseModel):
    """A plan for a RAG system. After running the plan, we should have either enough information to answer the user's original query, or enough information to form a new query plan."""
    reasoning: str = Field(
        description="Your step by step reasoning process on query formulation for semantic search."
    )
    decision: Literal["PLAN", "ANSWER"] = Field(description="Your decision whether to answer or replan for further queries")
    answer: str = Field(
        description="Answer that satisfies the initial query if the context is enough to answer it"
    )

class QueryPlanItemResult(Event):
    """The result of a query plan item"""

    query: str
    result: str


class ExecutedPlanEvent(Event):
    """The result of a query plan"""

    result: str