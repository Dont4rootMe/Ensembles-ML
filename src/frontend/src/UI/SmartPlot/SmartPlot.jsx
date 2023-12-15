import React, { useState } from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import Plot from 'react-plotly.js';

const SmartPlot = ({ data }) => {
    return (
        <Tabs
            defaultActiveKey="hider"
            id="uncontrolled-tab-example"
            className="mb-3"
            style={{ padding: '5px' }}
        >
            <Tab eventKey={'hider'} title={'Hide plot'}>

            </Tab>
            {
                Object.keys(data).map(key =>
                    <Tab eventKey={key} key={Math.floor(Math.random() * 100000)} title={key}>
                        <Plot
                            layout={{ width: '100%', title: `Метрика ${key}` }}
                            data={[{
                                x: [...Array(data[key]['train'].length).keys()],
                                y: data[key]['train'],
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: 'blue' },
                                name: 'train-metric'
                            }, {
                                x: [...Array(data[key]['test'].length).keys()],
                                y: data[key]['test'],
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: 'red' },
                                name: 'test-metric'
                            }]}
                        />
                    </Tab>
                )
            }

        </Tabs>
    )
}

export default SmartPlot