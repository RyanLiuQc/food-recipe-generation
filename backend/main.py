from fastapi import FastAPI
from pydantic import BaseModel
from inference import generate_recipe
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class RecipeRequest(BaseModel):
    ingredients: str   # comma separated string
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/Generator")
def generate(req: RecipeRequest):
    ing_list = req.ingredients.split(",")
    for i in ing_list:
        i.strip()
    result = generate_recipe(ingredients_list=ing_list)
    
    return {"recipe": result}