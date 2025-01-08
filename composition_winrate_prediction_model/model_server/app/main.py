from fastapi import FastAPI, HTTPException
from app.predictor import analyze_composition
from app.data_constants import CHAMP_ID_TO_NAME

app = FastAPI()

@app.get("/analyze_comp")
async def analyze_composition_endpoint(q: str):
    """
    Analyze champion composition based on the provided IDs.
    
    Args:
        q (str): Comma-separated list of champion IDs (e.g., "1,2,3,4,5,6,7,8,9,10").

    Returns:
        dict: Result from the analyze_composition function.
    """
    try:
        champ_ids_list = list(map(int, q.split(",")))
        
        # Input Sanity Check
        if len(champ_ids_list) != 10:
            raise HTTPException( status_code=400, detail="Exactly 10 champion IDs must be provided.")
        for champ_id in champ_ids_list:
            if champ_id not in CHAMP_ID_TO_NAME:
                raise HTTPException( status_code=400, detail="Invalid Champion ID")
            
        return {"result": analyze_composition(champ_ids_list)}
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input. Please provide a comma-separated list of integers. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
