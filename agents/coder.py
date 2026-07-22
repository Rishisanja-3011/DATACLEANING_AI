import ast
import json
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def _make_json_safe(data: Any) -> str:
    """
    Serialize Python data structures into a readable JSON string.

    default=str prevents serialization failures for values such as
    pandas/numpy-specific objects.
    """
    return json.dumps(
        data,
        indent=2,
        default=str
    )


def _clean_llm_code(code: str) -> str:
    """
    Remove markdown code fences if the LLM returns them despite
    being instructed not to.
    """
    code = code.strip()

    if code.startswith("```python"):
        code = code[len("```python"):]

    elif code.startswith("```"):
        code = code[3:]

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


def _validate_python_syntax(code: str) -> None:
    """
    Validate that generated code is syntactically valid Python.

    Raises SyntaxError if validation fails.
    """
    ast.parse(code)


def generate_preprocessing_code(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    report: Dict[str, Any]
) -> str:
    """
    Generate preprocessing code using Gemini.

    The Coder consumes:
    1. The Inspector's structured report.
    2. The Planner's structured preprocessing plan.

    The returned code must modify a DataFrame named `df`.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are the Coder Agent in an automated data preprocessing system.

Your responsibility is to generate safe, deterministic Python preprocessing
code from a structured dataset inspection report and preprocessing plan.

A pandas DataFrame named `df` already exists in the execution environment.

You MUST follow the preprocessing plan exactly.

SUPPORTED ACTIONS

Dataset-level actions:

1. remove_duplicates
   - Remove duplicate rows using df.drop_duplicates().

Column-level actions:

2. drop_column
   - Drop the specified column.

3. median_imputation
   - Use sklearn.impute.SimpleImputer(strategy="median").

4. most_frequent_imputation
   - Use sklearn.impute.SimpleImputer(strategy="most_frequent").

5. onehot_encoding
   - Use pandas.get_dummies().
   - Use drop_first=True.

6. label_encoding
   - Use sklearn.preprocessing.LabelEncoder.
   - Handle missing values before encoding when necessary.

7. clip_outliers
   - Calculate Q1 and Q3.
   - Calculate IQR.
   - Clip values to:
       Q1 - 1.5 * IQR
       Q3 + 1.5 * IQR

8. standard_scaling
   - Use sklearn.preprocessing.StandardScaler.

9. extract_datetime_features
   - Convert the column using pandas.to_datetime(errors="coerce").
   - Extract year, month, and day.
   - Create new columns using:
       <column>_year
       <column>_month
       <column>_day
   - Drop the original datetime column.

EXECUTION RULES

- Return ONLY executable raw Python code.
- Do NOT use markdown code fences.
- Do NOT provide explanations.
- Do NOT read files.
- Do NOT write files.
- Do NOT create a new source DataFrame.
- The existing variable `df` must contain the final cleaned DataFrame.
- Import only libraries required for preprocessing.
- Use pandas and scikit-learn.
- Do not use os, sys, subprocess, pathlib, shutil, socket, requests,
  or other system/network libraries.

IMPORTANT TRANSFORMATION RULE

Scikit-learn transformations on one column usually return a 2D array.

Therefore, when assigning the result to one DataFrame column, convert it
to one dimension.

Correct examples:

df["Age"] = imputer.fit_transform(df[["Age"]]).ravel()

df["Salary"] = scaler.fit_transform(df[["Salary"]]).ravel()

Do not assign a 2D array directly to a single DataFrame column.

ACTION ORDER

Respect the order of actions provided by the preprocessing plan.

If a column is dropped, do not perform later transformations on that column.

Generate straightforward and readable preprocessing code.
"""
        ),
        (
            "user",
            """
DATASET INSPECTION REPORT:

{inspection_report}


PREPROCESSING PLAN:

{preprocessing_plan}


Generate the Python preprocessing code.
"""
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "inspection_report": _make_json_safe(report),
        "preprocessing_plan": _make_json_safe(plan)
    })

    code = response.content

    if not isinstance(code, str) or not code.strip():
        raise ValueError(
            "Coder Agent received an empty response from the LLM."
        )

    code = _clean_llm_code(code)

    # Validate generated Python before sending it to the Reviewer.
    _validate_python_syntax(code)

    return code