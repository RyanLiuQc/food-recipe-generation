import { useState } from "react"

const Generator = () => {

    const [ingredients, setIngredients] = useState("chicken, spinach, cream, garlic")
    const [recipe, setRecipe] = useState("")

    const ready = async() => {
        try {
            const waiting = await fetch("http://127.0.0.1:8000/Generator", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ingredients: ingredients})
            })

            const result = await waiting.json()
            setRecipe(result.recipe)

        } catch (error){
            alert("oops that didn't work")
        }
    }

    return (
        <div className="generator_main">
            <div className="input_things">
                <p className="ing_instructions">Enter ingredients here:</p>
                <textarea 
                    className="ing_list" 
                    value={ingredients} 
                    onChange={(event) => setIngredients(event.target.value)}>
                </textarea>
                <button className="ready_btn" onClick={ready}>Ready to cook</button>
            </div>
            <div className="recipe_div">
                <p className="recipe">{recipe}</p>
            </div>
        </div>
    )
}

export default Generator