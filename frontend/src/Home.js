import { useNavigate } from "react-router-dom"
const Home = () => {
    const navigate = useNavigate();
    return (
        <div className="App">
            <header className="App-header">
                <div className='text_home'>
                    <p className="hook">
                        Generate a recipe with ingredients in <b>your</b> kitchen
                    </p>
                    <button className='get_started_btn' onClick={() => navigate("/Generator")}>
                        GET STARTED
                    </button>
                </div>
            </header>
        </div>
    )
}

export default Home;