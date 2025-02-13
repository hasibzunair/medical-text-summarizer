# Code for Medical Text Summarizer
# Author: Hasib Zunair

"""Medical Text Summarizer.

This is the code for the medical text summarizer API.
"""

import time
import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from predictor import summarize_notes

# set logger (datetime, level and message)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# init fastapi
app = FastAPI()


@app.post("/summarize")
async def summarize(request: Request):
    """Summarize notes based on the role with refere toe source."""
    try:
        data = await request.json()

        # read notes
        text = data.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text found.")
        # read custom role, default is physician
        clinical_role = data.get("clinical_role", "physician")

        start_time = time.time()
        # run inference
        summary, tokens, references = summarize_notes(text, clinical_role)
        duration = time.time() - start_time

        # log time and tokens
        logging.info(
            f"Summary of notes generated in {duration:.2f} seconds using {tokens} tokens."
        )

        return {
            "summary": summary,
            "tokens": tokens,
            "references": references,
            "processing_time": duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Check the status of the API endpoint."""
    return {"status": "this works"}


@app.post("/feedback")
async def feedback(request: Request):
    """Get feedback from the user."""
    try:
        data = await request.json()
        # read feednback
        feedback_text = data.get("feedback")

        if not feedback_text:
            raise HTTPException(status_code=400, detail="Feedback input is missing.")

        # log user feedback
        logging.info(f"Feedback: {feedback_text}")
        return {"message": "Got feedback."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # run
    uvicorn.run(app, host="0.0.0.0", port=8000)
