import { useState } from "react"

const Generator = () => {

    const [ingredients, setIngredients] = useState("chicken, spinach, cream, garlic")
    const ready = async() => {
        try {
            const waiting = fetch("http://localhost:5000/generate", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(ingredients)
            })

            const result = await waiting.json()

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
        </div>
    )
}

export default Generator