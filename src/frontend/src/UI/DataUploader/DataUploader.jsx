import React, {useState, useEffect} from "react";
import { Form, Button } from "react-bootstrap";
import RangeSlider from "react-bootstrap-range-slider";



const DataUploader = ({config}) => {
    const [dataSet, setDataSet] = useState(undefined);
    const [validationSet, setValidationSet] = useState(undefined);
    const [internal_config, ] = useState({
        autoValid: true,
        validSize: 30, 
        targetIndex: undefined
    })
    

    const _Validation_Type = ({}) => {
        const [autoValid, setAutovalid] = useState(true)
        const [validSize, setValidSize] = useState(30)

        useEffect(() => {
            internal_config.autoValid = autoValid
            internal_config.validSize = validSize
        }, [autoValid, validSize])

        return (
            <>
                <Form.Check
                    style={{fontSize: '0.8em'}}
                    type={'checkbox'}
                    label={`Атоматически определить тестовую подвыборку`}
                    id={`disabled-default-bootstrap`}
                    checked={autoValid}
                    onChange={e => setAutovalid(!autoValid)}
                />
                {autoValid ? <>
                    <Form.Text size="mm">{`процент на валидацию: ${validSize}%`}</Form.Text>
                    <RangeSlider 
                        max={50}
                        min={10}
                        variant='primary'
                        value={validSize}
                        tooltipLabel={currentValue => `${currentValue}%`}
                        onChange={e => setValidSize(e.target.value)}
                    />
                </> : 
                <Form.Group controlId="formFile" size="sm" style={{marginBottom: '5px'}}>
                    <Form.Label style={{marginBottom: '0px'}}>Выберете датасет для теста</Form.Label>
                    <Form.Control type="file" onChange={(e) => setValidationSet(e.target.files[0])} />
                </Form.Group>}
            </>
        )
    }

    const _Target_define = ({}) => {
        const [targetIndex, setTargetIndex] = useState(undefined)

        useEffect(() => {
            internal_config.targetIndex = targetIndex
        }, [targetIndex])

        return (
            <Form.Group>
                <Form.Text >Имя колонки таргета или его индекс</Form.Text>
                <Form.Control size="sm" type="text" placeholder="target" value={targetIndex} onChange={e => setTargetIndex(e.target.value)}/>
            </Form.Group>
        )
    } 

    return (
        <>
            <Form.Group controlId="formFile" size="sm" style={{marginBottom: '5px'}}>
                <Form.Label style={{marginBottom: '0px'}}>Выберете датасет для обучения</Form.Label>
                <Form.Control type="file" onChange={(e) => setDataSet(e.target.files[0])} />
            </Form.Group>
            <_Validation_Type />

            <_Target_define />

            <Button variant='success' 
                    style={{marginTop: '10px'}}
            >Обучить модель!</Button>
        </>
    )
}

export default DataUploader