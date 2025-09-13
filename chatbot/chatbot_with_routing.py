import os
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.workflow import (
    Workflow,
    StopEvent,
    StartEvent,
    Context,
    step,
    Event,
)
from llama_index.core.prompts import PromptTemplate
from .models import QueryPlanItem, QueryPlan, QueryPlanItemResult, ExecutedPlanEvent
os.environ["GOOGLE_API_KEY"] = "AIzaSyAkGA-LaMrWvXwyEbE5PZeeX6o2WsS2HP8"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
llm = GoogleGenAI(model="gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])


def read_docs_from_dir(input_dir: Path):
    md_files = input_dir.glob("**/*.md")
    return md_files


def build_vector_index(persist_dir: str):
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        return load_index_from_storage(storage_context)
    else:
        index_docs = []
        for doc in read_docs_from_dir(Path("./data/parsed")):
            with open(doc) as f:
                index_doc = Document(
                    text=f.read(),
                    metadata={"file_path": str(doc)}
                )
                index_docs.append(index_doc)

        pipeline = IngestionPipeline(
            name="document_parsing_pipeline",
            project_name="krisp_hackathon",
            transformations=[
                SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=embed_model
                )
            ]
        )
        nodes = pipeline.run(documents=index_docs)
        vector_index = VectorStoreIndex(nodes=nodes, use_async=True)
        vector_index.storage_context.persist(persist_dir)
        return vector_index

index_persist_path = "./storage/documents_index"
documents_index = build_vector_index(index_persist_path)
documents_query_engine = documents_index.as_query_engine(similarity_top_k=10, llm=llm)
documents_tool = QueryEngineTool(
    query_engine=documents_query_engine,
    metadata=ToolMetadata(
        name="documents_tool",
        description="Use this tool to execute semantic queries against document store."
    ),
)
all_tools = [documents_tool]

class QueryPlanningWorkflow(Workflow):
    planning_prompt = PromptTemplate(
        "Think step by step. Given an initial query, as well as information about the indexes you can query, return a plan for a RAG system.\n"
        "The plan should be a list of QueryPlanItem objects, where each object contains a query.\n"
        "The result of executing an entire plan should provide a result that is a substantial answer to the initial query, "
        "or enough information to form a new query plan.\n"
        "Sources you can query: {context}\n"
        "Initial query: {query}\n"
        "Plan:"
    )
    decision_prompt = PromptTemplate(
        "Given the following information, return a final response that satisfies the original query, or return 'PLAN' if you need to continue planning.\n"
        "Original query: {query}\n"
        "Current results: {results}\n"
    )

    @step
    async def planning_step(
        self, ctx: Context, ev: StartEvent | ExecutedPlanEvent
    ) -> QueryPlanItem | StopEvent:
        if isinstance(ev, StartEvent):
            # Initially, we need to plan
            query = ev.get("query")
            tools = ev.get("tools")
            await ctx.store.set("tools", {t.metadata.name: t for t in tools})
            await ctx.store.set("original_query", query)
            context_str = "\n".join(
                [
                    f"{i+1}. {tool.metadata.name}: {tool.metadata.description}"
                    for i, tool in enumerate(tools)
                ]
            )
            await ctx.store.set("context", context_str)
            
            # Use Gemini's structured output for the plan
            query_plan = await llm.astructured_predict(
                QueryPlan,
                self.planning_prompt,
                context=context_str,
                query=query,
            )
            ctx.write_event_to_stream(
                Event(msg=f"Planning step: {query_plan}")
            )
            num_items = len(query_plan.items)
            await ctx.store.set("num_items", num_items)
            for item in query_plan.items:
                ctx.send_event(item)
        else:
            # Decide if we need to replan or stop
            query = await ctx.store.get("original_query")
            current_results_str = ev.result
            decision = await llm.apredict(
                self.decision_prompt,
                query=query,
                results=current_results_str,
            )

            if "PLAN" in decision.upper():
                context_str = await ctx.store.get("context")
                query_plan = await llm.astructured_predict(
                    QueryPlan,
                    self.planning_prompt,
                    context=context_str,
                    query=query,
                )
                ctx.write_event_to_stream(
                    Event(msg=f"Re-Planning step: {query_plan}")
                )
                num_items = len(query_plan.items)
                await ctx.store.set("num_items", num_items)
                for item in query_plan.items:
                    ctx.send_event(item)
            else:
                return StopEvent(result=decision)

    @step(num_workers=4)
    async def execute_item(
        self, ctx: Context, ev: QueryPlanItem
    ) -> QueryPlanItemResult:
        tools = await ctx.store.get("tools")
        tool = tools[ev.name]
        ctx.write_event_to_stream(
            Event(
                msg=f"Querying tool {tool.metadata.name} with query: {ev.query}"
            )
        )
        result = await tool.acall(ev.query)
        ctx.write_event_to_stream(
            Event(msg=f"Tool {tool.metadata.name} returned: {result}")
        )
        return QueryPlanItemResult(query=ev.query, result=str(result))

    @step
    async def aggregate_results(
        self, ctx: Context, ev: QueryPlanItemResult
    ) -> ExecutedPlanEvent:
        num_items = await ctx.store.get("num_items")
        results = ctx.collect_events(ev, [QueryPlanItemResult] * num_items)
        if results is None:
            return
        aggregated_result = "\n------\n".join(
            [
                f"Query: {result.query}\nResult: {result.result}"
                for result in results
            ]
        )
        return ExecutedPlanEvent(result=aggregated_result)


async def main():
    workflow = QueryPlanningWorkflow(verbose=False, timeout=120)
    print("--- User Query 1: What is the main topic of the documents? ---")
    handler1 = workflow.run(
        query="How did Google Cloud perform in Q3 2023? What was its revenue and operating income?",
        tools=all_tools,
    )
    async for event in handler1.stream_events():
        if hasattr(event, "msg"):
            print(event.msg)
    result1 = await handler1
    print("\nChatbot Response:", str(result1))
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())