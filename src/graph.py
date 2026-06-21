"""Multi-agent orchestration.

Uses LangGraph if installed (makes the agent flow explicit, which is good for
the report/demo). Falls back to a plain sequential pipeline otherwise, so the
project runs even without langgraph.

Flow:  listener -> retriever -> risk -> planner
"""
from typing import TypedDict, Optional, List

from src.agents.listener import analyze
from src.agents.risk import evaluate
from src.agents.planner import plan

# Retriever is instantiated lazily (it needs the FAISS index on disk).
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.agents.retriever import Retriever
        _retriever = Retriever()
    return _retriever


class State(TypedDict):
    text: str
    analysis: Optional[dict]
    evidence: Optional[List[dict]]
    risk: Optional[dict]
    response: Optional[str]


def n_listen(s: State):
    return {"analysis": analyze(s["text"])}


def n_retrieve(s: State):
    a = s["analysis"]
    query = f"{a['category']} {a['clean_text']}"
    return {"evidence": _get_retriever().retrieve(query, k=4)}


def n_risk(s: State):
    a = s["analysis"]
    return {"risk": evaluate(a["clean_text"], a["distress"], a["is_abuse"])}


def n_plan(s: State):
    a, r = s["analysis"], s["risk"]
    return {"response": plan(s["text"], a["category"], a["distress"],
                             r["risk"], s["evidence"])}


# ---------------------------------------------------------------------------
# Build a compiled LangGraph app, or a sequential fallback with the same API.
# ---------------------------------------------------------------------------
def _build_langgraph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    g.add_node("listener", n_listen)
    g.add_node("retriever", n_retrieve)
    g.add_node("risk", n_risk)
    g.add_node("planner", n_plan)
    g.set_entry_point("listener")
    g.add_edge("listener", "retriever")
    g.add_edge("retriever", "risk")
    g.add_edge("risk", "planner")
    g.add_edge("planner", END)
    return g.compile()


class _SequentialApp:
    """Minimal drop-in replacement exposing .invoke() like a LangGraph app."""
    def invoke(self, state: State) -> State:
        state = {**state}
        state.update(n_listen(state))
        state.update(n_retrieve(state))
        state.update(n_risk(state))
        state.update(n_plan(state))
        return state


try:
    APP = _build_langgraph()
except Exception as exc:   # langgraph not installed -> fallback
    print(f"[graph] LangGraph unavailable ({exc}); using sequential pipeline.")
    APP = _SequentialApp()


def run(text: str) -> dict:
    final = APP.invoke({"text": text})
    a, r = final["analysis"], final["risk"]
    return {
        "category": a["category"],
        "distress": a["distress"],
        "emotion": a["emotion"],
        "is_abuse": a["is_abuse"],
        "risk": r["risk"],
        "urgent": r["urgent_note"],
        "evidence": final["evidence"],
        "response": final["response"],
    }


if __name__ == "__main__":
    import json, sys
    text = sys.argv[1] if len(sys.argv) > 1 else (
        "My manager keeps sending inappropriate messages and threatens "
        "my job if I report it.")
    out = run(text)
    print(json.dumps(out))
