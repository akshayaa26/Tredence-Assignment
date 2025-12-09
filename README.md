# CODE REVIEW MINI-AGENT
A minimal workflow engine with a Code Review Mini-Agent built using Python and FastAPI. Supports stateful execution, branching, looping, and tool-based modularity. Demonstrates backend engineering fundamentals with clean APIs and an extensible graph-based agent workflow.

# 🚀 How to Run

1)Install dependencies:
pip install -r requirements.txt

2)Start the FastAPI server:
uvicorn main:app --reload --port 8000

3)Open the API documentation:
http://127.0.0.1:8000/
(Auto-redirects to Swagger UI)

4)Use /graph/create to create a workflow graph.

5)Use /graph/run to execute the workflow with an initial state.

6)View results using /graph/state/{run_id}.

# ⚙️ What This Workflow Engine Supports

-Node-Based Execution: Each step is a Python function that reads/writes shared state.

-State Propagation: A dictionary flows through nodes and is updated at each step.

-Directed Graph Execution: Edges define the execution order between nodes.

-Branching: Conditional routing based on runtime values (e.g., quality score check).

-Looping: Nodes can be revisited until a stopping condition is met.

-Tool Registry: Nodes are pluggable functions registered dynamically.

-REST APIs: Create, run, and inspect workflows via FastAPI endpoints.

-Example Agent: Includes a full Code Review Mini-Agent demonstrating function extraction, complexity analysis, issue detection, and iterative refinement.

# 🔧 What I Would Improve With More Time 

-Async Execution: Convert node execution to async tasks with concurrency support.

-WebSocket Log Streaming: Live streaming of execution logs step-by-step.

-Persistent Storage: Store graphs and workflow runs in SQLite/PostgreSQL instead of memory.

-UI Dashboard: Build a small frontend to visualize graph execution and state transitions.

-Advanced Code Parsing: Replace regex-based extraction with Python AST for more accurate analysis.

-Plugin System: Allow external modules to register tools and custom workflows seamlessly.

-Error Handling & Metrics: Add structured logging, tracing, and metric collection for observability.
