import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/css/bootstrap.css'; // or include from a CDN
import 'react-bootstrap-range-slider/dist/react-bootstrap-range-slider.css';
import ToastList from './ToastFactory';
import { addDanger } from './ToastFactory';
import ModelSettings from './ModelSettings'
import ModelAudit from './ModelAudit';


function App() {
  const [historyOffTop, setHistoryOffTop] = useState(0)
  const [modelHistoryLine, setModelHistoryLine] = useState([])

  const addHistory = (history) => {
    setModelHistoryLine([...modelHistoryLine, {'key': historyOffTop, 'history': history}])
    setHistoryOffTop(historyOffTop + 1)
  }

  const deleteHistory = (key) => {
    let temp = []
    for (const hist of modelHistoryLine) {
      if (hist.key !== key) {
        temp.append(hist)
      }
    }

    setModelHistoryLine(temp)
  }


  return (
    <div style={{ height: '100%', width: '100%', paddingTop: '3%', paddingLeft: '7%', paddingRight: '7%' }}>
      <ToastList />
      <div style={{ width: '100%', display: 'grid', gridTemplateColumns: '3fr 7fr' }}>
        <ModelSettings addHistory={addHistory}/>
        <ModelAudit deleteHistory={deleteHistory}/>
      </div>
    </div>
  );
}

export default App;
