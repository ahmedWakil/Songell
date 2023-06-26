import { useEffect, useState } from 'react';
import PropTypes from "prop-types";
import './App.css';
import './generate';
import { sample } from './generate';

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
  //load data using fetch and useEffect hook to handle async calls we need to load this data
  //before all of the components mount
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/infrencing-model/encoding-data.json')
      .then((response) => { return response.json() })
      .then((json) => {
        const fetchedData = {}
        fetchedData.eos = json.eos;
        fetchedData.sos = json.sos;
        fetchedData.char_set = json.char_set;
        fetchedData.categories = json.categories;
        fetchedData.h_size = json.h_size;
        setData(fetchedData);
      });
  }, []);

  // states to keep track of the category and the generated names
  const [categoryIndex, setCategoryIndex] = useState(0)
  const [namesList, setNamesList] = useState(["", "", ""]);

  //handler for switching categories
  function categorySwitchHandler(index, direction) {
    if (direction === "left") {
      setCategoryIndex(index === 0 ? data.categories.length - 1 : (index - 1) % data.categories.length)
    } else if (direction === "right") {
      setCategoryIndex(index === data.categories.length - 1 ? 0 : (index + 1) % data.categories.length)
    }
  }

  //handler for the generate button, main task is get a number of names (default 3) and update the names list state
  async function generateHandler(categoryIndex, num = 3) {
    const sampledList = []

    for (let i = 0; i < num; i++) {
      const sampleInference = await sample(categoryIndex, data);
      sampledList.push(sampleInference)
    }

    console.log(`Category: ${data.categories[categoryIndex]}\nGenerated names: ${sampledList}`);
    setNamesList([...sampledList])
  }

  //List to contain the h3 elements that is suppoed to be displayed
  const namesHeaders = namesList.map((name, index) => (
    <h3 key={index}>{name}</h3>
  ));

  return (
    <main>
      <h1>Songell</h1>
      <h2>A Fantasy Name generator for your next RPG adventure</h2>
      {data && <CategorySelector
        categoryIndex={categoryIndex}
        categories={data.categories}
        categorySwitchHandler={categorySwitchHandler} />}

      <div className="card">
        <button onClick={() => generateHandler(categoryIndex)}>
          Generate
        </button>
      </div>

      <div className='line-container'>
        {namesHeaders}
      </div>
    </main>
  )
}

export default App
