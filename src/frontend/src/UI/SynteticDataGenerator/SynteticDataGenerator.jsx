import React, {useState} from "react";
import { Form, Button } from "react-bootstrap";
import RangeSlider from 'react-bootstrap-range-slider';

const SynteticDataGenerator = ({config}) => {
    const [sampleSize, setSampleSize] = useState(30000)
    const [featureSize, setFeatureSize] = useState(15)

    return (
        <>
            <Form.Group size="sm">
                <Form.Text size="mm">{`Количество объектов: ${sampleSize}`}</Form.Text>
                <RangeSlider 
                    max={100000}
                    min={1000}
                    variant='warning'
                    value={sampleSize}
                    onChange={e => setSampleSize(e.target.value)}
                />
            </Form.Group>
            <Form.Group size="sm">
                <Form.Text size="mm">{`Число признаков: ${featureSize}`}</Form.Text>
                <RangeSlider 
                    max={150}
                    min={5}
                    variant='warning'
                    value={featureSize}
                    onChange={e => setFeatureSize(e.target.value)}
                />
            </Form.Group>

            <Button variant='success' 
                    style={{marginTop: '10px'}}
            >Обучить модель!</Button>
        </>
    )
}

export default SynteticDataGenerator