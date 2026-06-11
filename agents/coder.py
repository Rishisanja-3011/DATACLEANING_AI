import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def generate_preprocessing_code(df, plan, report):
    """
    Agent 2 (The Coder): Writes Python code using Scikit-Learn to apply the correct preprocessing.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Python data scientist. 
You will be provided with information about a pandas DataFrame `df` and a preprocessing plan. 
Write Python code to clean the dataframe according to the plan. 
Use scikit-learn for imputation and scaling when necessary. 

CRITICAL INSTRUCTIONS:
- Return ONLY the raw python code snippet. 
- DO NOT use markdown formatting like ```python. 
- DO NOT add explanations.
- Make sure to import necessary scikit-learn modules (e.g., SimpleImputer, StandardScaler, LabelEncoder) at the top of the code.
- The code must modify `df` directly.
- DO NOT read any CSV file, `df` is already a variable in the environment.
"""),
        ("user", "Data shape: {shape}\nColumns: {columns}\nMissing values: {missing_values}\nPlan: {plan}\n\nWrite the python code.")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "shape": report.get("shape", df.shape),
        "columns": report.get("columns", df.dtypes.astype(str).to_dict()),
        "missing_values": report.get("missing_values", df.isnull().sum().to_dict()),
        "plan": plan
    })
    
    code = response.content.replace("```python", "").replace("```", "").strip()
    return code