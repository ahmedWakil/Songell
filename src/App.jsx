import { useState } from 'react';
import PropTypes from "prop-types";
import './App.css';
import './generate';
import { sample } from './generate';

const MODEL_DATA = {}

await fetch('/infrencing-model/encoding-data.json')
  .then((response) => response.json())
  .then((json) => {
    MODEL_DATA.eos = json.eos;
    MODEL_DATA.sos = json.sos;
    MODEL_DATA.char_set = json.char_set;
    MODEL_DATA.categories = json.categories;
    MODEL_DATA.h_size = json.h_size;
  });

function CategorySelector(props) {

  const categoryList = props.categories.map((category) => (
    <h3 key={category} className='category-header'>
      {category}
    </h3>
  ));

  return (
    <div className='category-selector-container'>
      <button className="selector-button" onClick={() => props.categorySwitchHandler(props.categoryIndex, "left")}>	&#60;</button>
      {categoryList[props.categoryIndex]}
      <button className="selector-button" onClick={() => props.categorySwitchHandler(props.categoryIndex, "right")}>&#62;</button>
    </div>
  )
}

CategorySelector.propTypes = {
  categoryIndex: PropTypes.number.isRequired,
  categories: PropTypes.arrayOf(PropTypes.string).isRequired,
  categorySwitchHandler: PropTypes.func.isRequired
}

function App() {

  const [categoryIndex, setCategoryIndex] = useState(0)

  function categorySwitchHandler(index, direction) {
    console.log(categoryIndex)
    if (direction === "left") {
      setCategoryIndex(index === 0 ? MODEL_DATA.categories.length - 1 : (index - 1) % MODEL_DATA.categories.length)
    } else if (direction === "right") {
      setCategoryIndex(index === MODEL_DATA.categories.length - 1 ? 0 : (index + 1) % MODEL_DATA.categories.length)
    }
  }

  async function generateHandler(categoryIndex) {
    const sampleList = [];

    for (let i = 0; i < 3; i++) {
      const sampleInference = await sample(categoryIndex, MODEL_DATA);
      sampleList.push(sampleInference);
    }

    console.log(sampleList)
  }

  return (
    <main>
      <h1>Songell</h1>
      <h2>A Fantasy Name generator for your next RPG adventure</h2>
      <CategorySelector
        categoryIndex={categoryIndex}
        categories={MODEL_DATA.categories}
        categorySwitchHandler={categorySwitchHandler} />

      <div className="card">
        <button onClick={() => generateHandler(categoryIndex)}>
          Generate
        </button>
      </div>

      <div className='line-container'>

      </div>
    </main>
  )
}

export default App
