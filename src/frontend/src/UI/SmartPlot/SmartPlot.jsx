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
                    <Tab eventKey={key} title={key}>
                        <Plot
                            layout={{ width: '100%', title: `Метрика ${key}` }}
                            data={[{
                                x: [...Array(data[key].length).keys()],
                                y: data[key],
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: { color: 'red' },
                            }]}
                        />
                    </Tab>
                )
            }

        </Tabs>
    )
}

export default SmartPlot