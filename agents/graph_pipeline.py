import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END

# =========================
# IMPORT AGENTS
# =========================
from inspector import inspect_data
from planner import create_plan
from coder import generate_preprocessing_code
from reviewer import execute_and_review

# =========================
# DEFINE SHARED STATE
# =========================
class GraphState(TypedDict):
    df: pd.DataFrame
    report: dict
    plan: dict
    python_code: str
    cleaned_df: pd.DataFrame
    review: dict

# =========================
# INSPECTOR NODE
# =========================
def inspector_node(state):
    print("\n========== RUNNING INSPECTOR ==========\n")
    report = inspect_data(state["df"])
    print(report)
    return {"report": report}

# =========================
# PLANNER NODE
# =========================
def planner_node(state):
    print("\n========== RUNNING PLANNER ==========\n")
    plan = create_plan(state["report"])
    print(plan)
    return {"plan": plan}

# =========================
# CODER NODE
# =========================
def coder_node(state):
    print("\n========== RUNNING CODER (LLM) ==========\n")
    
    python_code = generate_preprocessing_code(
        state["df"],
        state["plan"],
        state["report"]
    )
    
    print("\nGENERATED CODE:")
    print(python_code)
    
    return {"python_code": python_code}

# =========================
# REVIEWER NODE
# =========================
def reviewer_node(state):
    print("\n========== RUNNING REVIEWER ==========\n")
    
    review, cleaned_df = execute_and_review(
        state["df"],
        state["python_code"]
    )
    
    print(review)
    
    if review.get("execution_success"):
        print("\nNEW SHAPE:")
        print(cleaned_df.shape)
    else:
        print("\nEXECUTION FAILED:")
        print(review.get("error_message"))
        
    return {
        "review": review,
        "cleaned_df": cleaned_df
    }

# =========================
# BUILD GRAPH
# =========================
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("inspector", inspector_node)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

# Define Flow
workflow.set_entry_point("inspector")
workflow.add_edge("inspector", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", END)

# Compile Graph
app = workflow.compile()

if __name__ == "__main__":
    # =========================
    # LOAD DATASET
    # =========================
    try:
        df = pd.read_csv("electriccard.csv")
        
        # =========================
        # RUN GRAPH
        # =========================
        result = app.invoke({"df": df})
        
        # =========================
        # SAVE OUTPUT
        # =========================
        if result["review"].get("execution_success"):
            result["cleaned_df"].to_csv("cleaned_data.csv", index=False)
            print("\n✅ Cleaned dataset saved!")
        else:
            print("\n❌ Pipeline failed to clean dataset.")
            
    except FileNotFoundError:
        print("electriccard.csv not found. Are you in the root directory?")