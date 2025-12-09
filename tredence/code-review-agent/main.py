# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
import re
import time

app = FastAPI(title="Minimal Workflow Engine - Code Review Mini-Agent")

# -------------------------
# In-memory stores
# -------------------------
GRAPHS: Dict[str, Dict[str, Any]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Pydantic schemas
# -------------------------
class GraphCreate(BaseModel):
    nodes: List[str]  # list of node names (must be registered)
    edges: Dict[str, Union[str, Dict[str, str]]]  # next mapping; can be simple or conditional
    start_node: str

class GraphRunRequest(BaseModel):
    graph_id: str
    initial_state: Dict[str, Any]
    threshold: Optional[int] = 80  # quality threshold to stop the loop

class GraphCreateResponse(BaseModel):
    graph_id: str

class RunResponse(BaseModel):
    run_id: str
    final_state: Dict[str, Any]
    log: List[str]

# -------------------------
# Tool registry & helpers
# -------------------------
TOOLS: Dict[str, Any] = {}

def register_tool(name):
    def decorator(fn):
        TOOLS[name] = fn
        return fn
    return decorator

# -------------------------
# Node implementations (simple heuristics)
# -------------------------
@register_tool("extract_functions")
def extract_functions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract function definitions by a simple regex for 'def <name>('.
    Populates state["functions"] as list of {'name':..., 'code': '...'}
    """
    code = state.get("code", "")
    func_pattern = re.compile(r"def\s+([A-Za-z_]\w*)\s*\((.*?)\):")
    functions = []
    # naive split by lines to capture function body heuristically
    lines = code.splitlines()
    for i, line in enumerate(lines):
        m = func_pattern.search(line)
        if m:
            name = m.group(1)
            # capture following indented block as body (naive)
            body_lines = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith(" ") or lines[j].startswith("\t")):
                body_lines.append(lines[j])
                j += 1
            functions.append({"name": name, "args": m.group(2).strip(), "body": "\n".join(body_lines)})
    state["functions"] = functions
    state.setdefault("log", []).append(f"extract_functions: found {len(functions)} function(s).")
    return state

@register_tool("calculate_complexity")
def calculate_complexity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very rough complexity metric:
      - base complexity = number of lines in each function
      - +1 for each 'if','for','while','try' inside body
    Normalized as avg complexity per function.
    """
    funcs = state.get("functions", [])
    if not funcs:
        state["complexity_score"] = 0
        state.setdefault("log", []).append("calculate_complexity: no functions found, complexity=0.")
        return state

    total = 0
    for f in funcs:
        body = f.get("body", "")
        lines = [ln for ln in body.splitlines() if ln.strip() != ""]
        line_count = len(lines)
        control_hits = sum(1 for kw in ("if ", "for ", "while ", "try:", "except", "with ") if kw in body)
        score = line_count + control_hits
        f["complexity"] = score
        total += score

    avg = total / len(funcs)
    state["complexity_score"] = round(avg, 2)
    state.setdefault("log", []).append(f"calculate_complexity: avg complexity={state['complexity_score']}.")
    return state

@register_tool("detect_issues")
def detect_issues(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based detections:
     - TODO/FIXME comments
     - Very long lines (>120)
     - Functions with too many args (>4)
    Returns list of issue descriptions.
    """
    code = state.get("code", "")
    issues = []
    # TODO/FIXME
    for i, line in enumerate(code.splitlines(), 1):
        if "TODO" in line or "FIXME" in line:
            issues.append({"type": "todo", "line": i, "detail": line.strip()})
        if len(line) > 120:
            issues.append({"type": "long_line", "line": i, "detail": f"len={len(line)}"})
    # function arg counts
    for f in state.get("functions", []):
        args = f.get("args", "")
        if args.strip() != "":
            arg_count = len([a for a in args.split(",") if a.strip() != ""])
        else:
            arg_count = 0
        if arg_count > 4:
            issues.append({"type": "many_args", "function": f["name"], "detail": f"args={arg_count}"})
    state["issues"] = issues
    state.setdefault("log", []).append(f"detect_issues: detected {len(issues)} issue(s).")
    return state

@register_tool("suggest_improvements")
def suggest_improvements(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create simple suggestions:
     - for each issue, add a suggestion string
     - for high complexity, suggest splitting function
    """
    suggestions = state.get("suggestions", [])
    issues = state.get("issues", [])
    for issue in issues:
        t = issue.get("type")
        if t == "todo":
            suggestions.append(f"Remove TODO/FIXME or implement at line {issue.get('line')}.")
        elif t == "long_line":
            suggestions.append(f"Wrap long line at {issue.get('line')} (length {issue.get('detail')}).")
        elif t == "many_args":
            suggestions.append(f"Refactor function {issue.get('function')} to reduce args ({issue.get('detail')}).")
        else:
            suggestions.append(f"Investigate issue: {issue}")

    # complexity suggestions
    avg_complex = state.get("complexity_score", 0)
    if avg_complex >= 20:
        suggestions.append("High average complexity: consider splitting large functions into smaller units.")
    elif avg_complex >= 10:
        suggestions.append("Moderate complexity: review logic for simplification.")
    else:
        suggestions.append("Complexity acceptable.")

    state["suggestions"] = suggestions
    state.setdefault("log", []).append(f"suggest_improvements: added {len(suggestions)} suggestion(s).")
    return state

@register_tool("compute_quality")
def compute_quality(state: Dict[str, Any], threshold: int = 80) -> Dict[str, Any]:
    """
    Heuristic to compute a quality score in [0,100]:
    - base 100
    - subtract complexity_score * 2
    - subtract issues_count * 5
    clipped to 0..100
    """
    complexity = state.get("complexity_score", 0)
    issues = state.get("issues", [])
    score = 100 - (complexity * 2) - (len(issues) * 5)
    score = max(0, min(100, int(score)))
    state["quality_score"] = score
    state.setdefault("log", []).append(f"compute_quality: quality_score={score}. threshold={threshold}")
    return state

# -------------------------
# Graph engine
# -------------------------
def validate_graph(graph: Dict[str, Any]):
    # nodes must be registered
    for node in graph["nodes"]:
        if node not in TOOLS:
            raise HTTPException(status_code=400, detail=f"Node '{node}' is not a registered tool.")
    if graph["start_node"] not in graph["nodes"]:
        raise HTTPException(status_code=400, detail="start_node must be in nodes list.")

def run_graph_sync(graph: Dict[str, Any], initial_state: Dict[str, Any], threshold: int = 80) -> Dict[str, Any]:
    """
    Runs the graph in a simple synchronous manner and returns final state and logs.
    The graph.edges is a mapping node->next where next can be:
      - a string: next node
      - a dict with keys "true" and "false" used for branching decisions in node 'loop_check'
    We'll use a convention: graphs may include a 'loop_check' node that decides routing.
    """
    state = dict(initial_state)  # shallow copy
    state.setdefault("log", [])
    run_log = []
    current = graph["start_node"]
    visited = 0
    max_steps = 500  # guard against infinite loops

    while current and visited < max_steps:
        visited += 1
        run_log.append(f"Running node: {current}")
        state.setdefault("last_node", current)

        # special handling: compute_quality receives threshold param
        if current == "compute_quality" or current == "loop_check":
            state = TOOLS["compute_quality"](state, threshold)
        else:
            # call registered tool
            state = TOOLS[current](state)

        # decide next
        next_spec = graph["edges"].get(current)
        if next_spec is None:
            # terminated
            run_log.append(f"No outgoing edge from node '{current}'. Ending.")
            current = None
            break

        if isinstance(next_spec, str):
            next_node = next_spec
        elif isinstance(next_spec, dict):
            # expected: {"cond": "quality_score >= threshold", "true": "nodeA", "false":"nodeB"}
            cond = next_spec.get("cond")
            true_node = next_spec.get("true")
            false_node = next_spec.get("false")
            # we only support a specific condition token for simplicity:
            # "quality_below_threshold" : true -> go to true_node else false_node
            if cond == "quality_below_threshold":
                q = state.get("quality_score", 0)
                if q < threshold:
                    next_node = true_node
                else:
                    next_node = false_node
            else:
                # unknown conditional - stop
                run_log.append(f"Unknown conditional for node '{current}': {cond}. Stopping.")
                current = None
                break
        else:
            run_log.append(f"Invalid edge specification from '{current}'. Stopping.")
            current = None
            break

        run_log.append(f"Transition: {current} -> {next_node}")
        current = next_node

    # merge state log with run_log
    final_log = state.get("log", []) + run_log
    state["execution_log"] = final_log
    return state

# -------------------------
# Example graph builder: Code Review Mini-Agent
# -------------------------
def build_code_review_graph() -> Dict[str, Any]:
    """
    Nodes used:
      - extract_functions
      - calculate_complexity
      - detect_issues
      - suggest_improvements
      - compute_quality  (used as loop_check node)
    Edges:
      extract_functions -> calculate_complexity -> detect_issues -> suggest_improvements -> loop_check
      loop_check (conditional) -> if quality_below_threshold: suggest_improvements (to allow iterative improvement)
                          -> else: end (no next)
    """
    nodes = [
        "extract_functions",
        "calculate_complexity",
        "detect_issues",
        "suggest_improvements",
        "compute_quality",  # acts as loop_check
    ]
    edges = {
        "extract_functions": "calculate_complexity",
        "calculate_complexity": "detect_issues",
        "detect_issues": "suggest_improvements",
        "suggest_improvements": "compute_quality",
        # conditional: if quality below threshold -> go back to suggest_improvements, else -> terminate (no next)
        "compute_quality": {"cond": "quality_below_threshold", "true": "suggest_improvements", "false": None}
    }
    return {"nodes": nodes, "edges": edges, "start_node": "extract_functions"}

# -------------------------
# API endpoints
# -------------------------
@app.post("/graph/create", response_model=GraphCreateResponse)
def create_graph(payload: Optional[GraphCreate] = None):
    """
    Create a graph. If payload is None, create the example Code Review graph.
    """
    if payload:
        graph = payload.dict()
    else:
        graph = build_code_review_graph()

    # Validate nodes are registered
    try:
        validate_graph(graph)
    except HTTPException as e:
        raise e

    graph_id = str(uuid4())
    GRAPHS[graph_id] = graph
    return {"graph_id": graph_id}

@app.post("/graph/run", response_model=RunResponse)
def run_graph(req: GraphRunRequest):
    # fetch graph
    graph = GRAPHS.get(req.graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail="graph_id not found")

    # run synchronously
    t0 = time.time()
    final_state = run_graph_sync(graph, req.initial_state, threshold=req.threshold)
    t1 = time.time()
    run_id = str(uuid4())
    RUNS[run_id] = {
        "graph_id": req.graph_id,
        "state": final_state,
        "created_at": t0,
        "finished_at": t1,
    }
    # Build a concise log for return (limit size for safety)
    log = final_state.get("execution_log", [])
    # truncate long logs if needed
    if len(log) > 500:
        log = log[-500:]
    return {"run_id": run_id, "final_state": final_state, "log": log}

@app.get("/graph/state/{run_id}")
def get_state(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {"run_id": run_id, "state": run["state"], "graph_id": run["graph_id"]}

@app.get("/graphs")
def list_graphs():
    return {"graphs": list(GRAPHS.keys())}

@app.get("/runs")
def list_runs():
    return {"runs": list(RUNS.keys())}
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/docs")
