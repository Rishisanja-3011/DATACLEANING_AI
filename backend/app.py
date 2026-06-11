from fastapi import FastAPI, UploadFile
import pandas as pd

app = FastAPI()

@app.post("/upload")
async def upload_csv(file: UploadFile):

    df = pd.read_csv(file.file)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns)
    }