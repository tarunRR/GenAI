from fastapi import FastAPI
import time

app = FastAPI()

@app.post("/generate")
def generate_response(prompt: str):
    start = time.time()
    output = model.generate(...)
    latency = time.time() - start
    
    logging.info(f"Latency: {latency}")
    return {"response": output}