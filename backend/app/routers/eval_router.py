from fastapi import APIRouter
router = APIRouter()

@router.get("/status")
async def eval_status():
    return {"status": "Evaluation module ready. POST /api/eval/run to start."}
