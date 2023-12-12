import React, {useState, useEffect} from "react";
import { Form, Button, Dropdown } from "react-bootstrap";
import RangeSlider from 'react-bootstrap-range-slider';
import { call_post, BadResponse } from "../../CALLBACKS";
import { addDanger, addMessage, addSuccess } from "../../ToastFactory";

const SynteticDataGenerator = ({config, addHistory}) => {
    const [sampleSize, setSampleSize] = useState(30000)
    const [featureSize, setFeatureSize] = useState(15)
    const [validationSplitter, setValidationSplitter] = useState('30')

    useEffect(() => {
        if (!config) {return;}
        config.synt_prefs = {
            sample_size: sampleSize,
            feature_size: featureSize,
            validation_percent: validationSplitter
        }
    }, [config, sampleSize, featureSize, validationSplitter])


    const trainModel = async (trace) => {

        console.log(config)
        if (config.estimators.length <= 0 ) {
            addDanger('Число деревьев', 'Задайте явным образом параметр модели') 
            return
        }
        if (config.model === 'grad-boosting' && config.learningRate.length <= 0) {
            addDanger('Learning rate', 'Задайте явным образом learning rate') 
            return
        }
        
        addMessage('Обучение модели', 'Обучение модели учпешно началось')

        let history = {dataset: 'synt',
                       trace: trace, 
                       sample_size: sampleSize, 
                       feature_size: featureSize, 
                       validation_percent: validationSplitter,
                       config: {...config}
        }

        const reply = await call_post('http://localhost:8000/syntet-train', config, {trace: trace})
        if (reply instanceof BadResponse) {
            addDanger('Обучение модели', 'Что-то пошло не так')
        } else {
            addSuccess('Обучение модели', 'Модель обучилась успешно')
            addHistory({...history, ...reply.data})
        }
    }

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

            <Form.Group size="sm">
                <Form.Text size="mm">{`Доля на тест: ${validationSplitter}%`}</Form.Text>
                <RangeSlider 
                    max={50}
                    min={10}
                    variant='primary'
                    value={validationSplitter}
                    tooltipLabel={currentValue => `${validationSplitter}%`}
                    onChange={e => setValidationSplitter(e.target.value)}
                />
            </Form.Group>

            <Dropdown >
                <Dropdown.Toggle variant="success" style={{marginTop: '10px'}}>
                    Обучить модель!
                </Dropdown.Toggle>

                <Dropdown.Menu>
                    <Dropdown.Item href="#/action-historic-on" 
                        onClick={() => trainModel(true)}
                    >Показать историю</Dropdown.Item>
                    <Dropdown.Item href="#/action-historic-off"
                        onClick={() => trainModel(false)}
                    >Без истории</Dropdown.Item>
                </Dropdown.Menu>
            </Dropdown>
        </>
    )
}

export default SynteticDataGenerator