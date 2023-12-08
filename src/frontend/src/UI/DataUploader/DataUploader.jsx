import React, {useState, useEffect} from "react";
import { Form, Button, InputGroup, Dropdown, DropdownButton } from "react-bootstrap";
import RangeSlider from "react-bootstrap-range-slider";



const DataUploader = ({config}) => {
    const [internal_config, ] = useState({
        autoValid: true,
        validSize: 30, 
        targetIndex: undefined
    })

    const [dataSet, setDataSet] = useState(undefined);
    const [validationSet, setValidationSet] = useState(undefined);
    const [targetIndex, setTargetIndex] = useState(undefined)


    const [targetList, setTargetList] = useState([])
    const [autoValid, setAutovalid] = useState(true)
    const [validSize, setValidSize] = useState(30)

    useEffect(() => {
        internal_config.targetIndex = targetIndex
    }, [targetIndex])

    useEffect(() => {
        internal_config.autoValid = autoValid
        internal_config.validSize = validSize
    }, [autoValid, validSize])

    
    const _get_target_list = (delimiter) => {
        if (!dataSet) {return;}

        var reader = new FileReader();
        reader.onload = function() {
            const text = this.result;

            var line = text.split('\n')[0];
            setTargetList(line.split(delimiter));
        };
        reader.readAsText(dataSet);
    }
    
    return (
        <>
            <Form.Group controlId="formFile" size="sm" style={{marginBottom: '5px'}}>
                <Form.Label style={{marginBottom: '0px'}}>Выберете датасет для обучения</Form.Label>
                <Form.Control type="file" onChange={(e) => setDataSet(e.target.files[0])} />
            </Form.Group>
            
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

            <Form>
                <Form.Text>Выберите поле target</Form.Text>
                <InputGroup className="mb-3">
                    <DropdownButton
                    variant="outline-secondary"
                    title="del."
                    id="input-group-dropdown-1"
                    >
                        <Dropdown.Item onClick={() => _get_target_list(',')}>delimiter:   ","</Dropdown.Item>
                        <Dropdown.Item onClick={() => _get_target_list(';')}>delimiter:   ";"</Dropdown.Item>
                        <Dropdown.Item onClick={() => _get_target_list('\t')}>delimiter:   "\t"</Dropdown.Item>

                    </DropdownButton>
                    <Form.Select aria-label="Text input with dropdown button" 
                        onChange={(e) => setTargetIndex(e.target.value)}
                    >
                        {targetList.map(item => 
                            <option key={item} value={item}>
                                {item}
                            </option>)}
                    </Form.Select>
                </InputGroup>
            </Form>

            <Button variant='success' 
                    style={{marginTop: '10px'}}
            >Обучить модель!</Button>
        </>
    )
}

export default DataUploader