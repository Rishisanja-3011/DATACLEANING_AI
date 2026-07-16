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
- When assigning a transformed single column back to `df`, ensure the value is 1D. For example:
  `df['Data_value'] = imputer_mean.fit_transform(df[['Data_value']])[:, 0]` 
  or `df['Series_title_3'] = imputer_most_frequent.fit_transform(df[['Series_title_3']]).ravel()`.
- For a one-column transformation, always use `[:, 0]` or `.ravel()` after `fit_transform` before assignment to `df[column]`.
- If the plan specifies "onehot_encoding", use `df = pd.get_dummies(df, columns=[...], drop_first=True)`.
- If the plan specifies "extract_datetime_features", convert the column using `pd.to_datetime()`, extract year, month, and day into new columns, and drop the original date column.
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