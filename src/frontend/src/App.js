import React, { useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import ToastList from './ToastFactory';
import { addDanger } from './ToastFactory';
import ModelSettings from './ModelSettings'


function App() {
  return (
    <div style={{ height: '100%', width: '100%', paddingTop: '10%', paddingLeft: '5%' }}>
      <ToastList />
      <div style={{ width: '100%', display: 'grid', gridTemplateColumns: '3fr 7fr' }}>
        <ModelSettings />
      </div>
    </div>
  );
}

export default App;
