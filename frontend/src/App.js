import './App.css';
import { Route, Routes } from "react-router-dom";
import Home from './Home';
import Generator from './Generator';


const App = () => {
  
  return (
    <Routes>
      <Route path='/' element={<Home />}></Route>
      <Route path='/Generator' element={<Generator />}></Route>
    </Routes>


  );
}

export default App;
