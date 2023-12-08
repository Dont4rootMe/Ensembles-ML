import React, { useState, useEffect } from 'react'
import { Form, FloatingLabel } from 'react-bootstrap'
import ModelParams from './UI/ModelParams/ModelParams'
import DataSetDefiner from './DataSetDefiner'

const ModelSettings = ({addHistory}) => {

    const [config, setConfig] = useState({
        model: 'random-forest'
    })

    return (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h1 style={{ fontSize: '2.3em', }}>Настройка модели</h1>
            <FloatingLabel controlId="SelectModelType" label="Выберете модель">
                <Form.Control
                    as="select"
                    defaultValue={'random-forest'}
                    onChange={e => setConfig({ ...config, model: e.target.value })}
                    style={{ fontSize: '1.1em' }}
                >
                    <option key={'random-forest'} value={'random-forest'}>
                        Random forest
                    </option>
                    <option key={'grad-boosting'} value={'grad-boosting'}>
                        Gradient Boosting
                    </option>
                </Form.Control>
            </FloatingLabel>
            <ModelParams modelType={config['model']} config={config} />
            <DataSetDefiner config={config} addHistory={addHistory} />
        </div>
    )
}

export default ModelSettings