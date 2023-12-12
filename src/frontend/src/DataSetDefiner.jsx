import React, { useState } from "react";
import { Form, Row } from "react-bootstrap";
import { FloatingLabel } from "react-bootstrap";

import DataUploader from "./UI/DataUploader/DataUploader";
import SynteticDataGenerator from "./UI/SynteticDataGenerator/SynteticDataGenerator";

const DataSetDefiner = ({ config, addHistory }) => {
    const [dataSetType, setDataSetType] = useState('synt')

    return (
        <Form.Group style={{
            dislpay: 'flex', flexDirection: 'column',
            marginTop: '5px', paddingTop: '5px', padding: '8px',
            background: 'rgb(0,0,0,0.05)', borderRadius: '5px'
        }}>

            <FloatingLabel controlId="SelectModelType" label="Данные запуска" style={{ marginBottom: '0.5em' }}>
                <Form.Control
                    as="select"
                    defaultValue={dataSetType}
                    onChange={e => setDataSetType(e.target.value)}
                    style={{ fontSize: '1.1em' }}
                >
                    <option key={'synt'} value={'synt'}>
                        Синтетический тест
                    </option>
                    <option key={'own-set'} value={'own-set'}>
                        Тест на пользовательских данных
                    </option>
                </Form.Control>
            </FloatingLabel>

            {dataSetType === 'synt' ? <SynteticDataGenerator config={config} addHistory={addHistory} />
                : <DataUploader config={config} addHistory={addHistory} />}
        </Form.Group>
    );
}

export default DataSetDefiner