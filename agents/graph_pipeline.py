import pandas as pd

from typing import Any, Dict, TypedDict
from langgraph.graph import StateGraph, END

from agents.inspector import inspect_data
from agents.planner import create_plan
from agents.coder import generate_preprocessing_code
from agents.reviewer import execute_and_review


# =========================================================
# SHARED GRAPH STATE
# =========================================================

class GraphState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph agents.
    """

    # Input
    df: pd.DataFrame

    # Inspector output
    report: Dict[str, Any]

    # Planner output
    plan: Dict[str, Any]

    # Coder output
    python_code: str

    # Reviewer outputs
    cleaned_df: pd.DataFrame
    review: Dict[str, Any]


# =========================================================
# INSPECTOR NODE
# =========================================================

def inspector_node(state: GraphState) -> Dict[str, Any]:

    print("\n========== RUNNING INSPECTOR ==========\n")

    report = inspect_data(state["df"])

    print("Dataset shape:", report["shape"])
    print("Duplicate rows:", report["duplicate_rows"])

    return {
        "report": report
    }


# =========================================================
# PLANNER NODE
# =========================================================

def planner_node(state: GraphState) -> Dict[str, Any]:

    print("\n========== RUNNING PLANNER ==========\n")

    plan = create_plan(state["report"])

    print("Dataset actions:")
    print(plan["dataset_actions"])

    print("\nColumns requiring actions:")
    print(list(plan["column_actions"].keys()))

    return {
        "plan": plan
    }


# =========================================================
# CODER NODE
# =========================================================

def coder_node(state: GraphState) -> Dict[str, Any]:

    print("\n========== RUNNING CODER (LLM) ==========\n")

    python_code = generate_preprocessing_code(
        state["df"],
        state["plan"],
        state["report"]
    )

    print("\nGENERATED CODE:\n")
    print(python_code)

    return {
        "python_code": python_code
    }


# =========================================================
# REVIEWER NODE
# =========================================================

def reviewer_node(state: GraphState) -> Dict[str, Any]:

    print("\n========== RUNNING REVIEWER ==========\n")

    review, cleaned_df = execute_and_review(
        state["df"],
        state["python_code"]
    )

    print("Execution success:",
          review["execution_success"])

    print("Validation success:",
          review["validation_success"])

    print("Pipeline success:",
          review["pipeline_success"])

    print("Quality score:",
          review["quality_score"])

    if review["pipeline_success"]:

        print("\nOriginal shape:",
              review["original_shape"])

        print("Final shape:",
              review["final_shape"])

    else:

        print("\nPIPELINE REVIEW FAILED")

        if review.get("error_message"):
            print(
                "Error:",
                review["error_message"]
            )

    return {
        "review": review,
        "cleaned_df": cleaned_df
    }


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(GraphState)

workflow.add_node(
    "inspector",
    inspector_node
)

workflow.add_node(
    "planner",
    planner_node
)

workflow.add_node(
    "coder",
    coder_node
)

workflow.add_node(
    "reviewer",
    reviewer_node
)


# =========================================================
# DEFINE GRAPH FLOW
# =========================================================

workflow.set_entry_point("inspector")

workflow.add_edge(
    "inspector",
    "planner"
)

workflow.add_edge(
    "planner",
    "coder"
)

workflow.add_edge(
    "coder",
    "reviewer"
)

workflow.add_edge(
    "reviewer",
    END
)


# =========================================================
# COMPILE GRAPH
# =========================================================

app = workflow.compile()


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    dataset_path = "electriccard.csv"

    try:

        # -------------------------
        # Load Dataset
        # -------------------------

        df = pd.read_csv(dataset_path)

        print("\n====================================")
        print("DATACLEANING_AI")
        print("====================================")

        print("\nDataset loaded successfully.")

        print("Shape:", df.shape)

        # -------------------------
        # Execute Graph
        # -------------------------

        result = app.invoke({
            "df": df
        })

        # -------------------------
        # Save Successful Result
        # -------------------------

        if result["review"]["pipeline_success"]:

            result["cleaned_df"].to_csv(
                "cleaned_data.csv",
                index=False
            )

            print(
                "\n[SUCCESS] Cleaned dataset saved "
                "as cleaned_data.csv"
            )

        else:

            print(
                "\n[FAILED] Pipeline did not pass "
                "Reviewer validation."
            )

    except FileNotFoundError:

        print(
            f"\n[ERROR] Dataset '{dataset_path}' "
            "was not found."
        )

    except Exception as exc:

        print(
            "\n[ERROR] Pipeline crashed:"
        )

        print(str(exc))