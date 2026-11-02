from fastapi import FastAPI

app = FastAPI(title="MetroGuard AI API")

@app.get("/")
def read_root():
    return {"status": "MetroGuard AI API is operational"}
