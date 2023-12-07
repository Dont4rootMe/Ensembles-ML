import React, { useState, useEffect } from "react";
import InputGroupText from "react-bootstrap/esm/InputGroupText";
import { InputGroup, Form } from "react-bootstrap";
import { addMessage } from "../../ToastFactory";

let lastKeyDownMesage = Date.now() - 5000

const ModelParams = ({ modelType, config }) => {

    const checkNumericInput = (e, value, dotAcceptable=false) => {
        if (e.nativeEvent.inputType === 'deleteContentBackward') {
            return value.substring(0, value.length - 1)
        }
        if (e.nativeEvent.inputType === 'insertText' && 
            ((e.nativeEvent.data >= '0' && e.nativeEvent.data <= '9') || 
             (dotAcceptable && e.nativeEvent.data === '.'))) {
            return value + e.nativeEvent.data
        }
        if (Date.now() - lastKeyDownMesage > 5000) {
            addMessage('Задача числа деревьев', 'Число можно задавать лишь числами')
            lastKeyDownMesage = Date.now()
        }
        return value

    }


    const _Settings = ({type}) => {
        const [estimators, setEstimators] = useState('100')

        const [depth, setDepth] = useState('100')
        const [useMaxDepth, setUseMaxDepth] = useState(false)

        const [fetSubsample, setFetSubsample] = useState('')

        const [useRandomSplit, setUseRandomSplit] = useState(false)

        const [bootstrapCoef, setBootstrapCoef] = useState('')
        const [useBootstraping, setUseBootstraping] = useState(false)

        const [randomState, setRandomState] = useState('42')

        const [learningRate, setLearningRate] = useState('0.1')

        useEffect(() => {
            if (config) {
                config.estimators = estimators
                config.depth = useMaxDepth ? depth : undefined
                config.fetSubsample = fetSubsample === '' ? 1/3 : fetSubsample
                config.useRandomSplit = useRandomSplit
                config.bootstrapCoef = useBootstraping ? bootstrapCoef : undefined
                config.randomState = randomState === '' ? undefined : randomState
                config.learningRate = type==='grad-boosting' ? learningRate : undefined
            }
            console.log(config)
        }, [config, estimators, depth, useMaxDepth, fetSubsample, useRandomSplit, bootstrapCoef, useBootstraping, randomState, learningRate])

        return (
            <>
               <InputGroup size="sm" className="mb-3">
                    <InputGroup.Text id="inputGroup-sizing-sm">Число деревьев</InputGroup.Text>
                    <Form.Control
                        aria-label="Small"
                        aria-describedby="inputGroup-sizing-sm"
                        onChange={(e) => setEstimators(checkNumericInput(e, estimators))}
                        value={estimators}
                    />
                </InputGroup>

                <InputGroup size="sm" style={{marginBottom: '5px'}}>
                    <InputGroup.Text id="inputGroup-sizing-sm">Глубина деревьев</InputGroup.Text>
                    <Form.Control
                        aria-label="Small"
                        aria-describedby="inputGroup-sizing-sm"
                        disabled={!useMaxDepth}
                        onChange={(e) => setDepth(checkNumericInput(e, depth))}
                        value={depth}
                    />
                </InputGroup>
                <Form.Check
                    style={{fontSize: '0.8em'}}
                    type={'checkbox'}
                    label={`Не задавать максимальную глубину деревьев`}
                    id={`disabled-default-maxdepth`}
                    checked={!useMaxDepth}
                    onChange={() => setUseMaxDepth(!useMaxDepth)}
                />

                <InputGroup size="sm" style={{marginTop: '0.6rem'}}>
                    <InputGroup.Text id="inputGroup-sizing-sm">Число признаков</InputGroup.Text>
                    <Form.Control
                        aria-label="Small"
                        aria-describedby="inputGroup-sizing-sm"
                        placeholder="использовать 1/3"
                        onChange={(e) => setFetSubsample(checkNumericInput(e, fetSubsample, true))}
                        value={fetSubsample}
                    />
                </InputGroup>

                <Form.Check 
                    style={{marginTop: '0.6rem'}}
                    type="switch"
                    id="switch-strategy"
                    label="Стратегия ветвления random"
                    checked={useRandomSplit}
                    onChange={() => setUseRandomSplit(!useRandomSplit)}
                />

                <InputGroup size="sm" style={{marginBottom: '5px'}}>
                    <InputGroup.Text id="inputGroup-sizing-sm">Глубина деревьев</InputGroup.Text>
                    <Form.Control
                        aria-label="Small"
                        aria-describedby="inputGroup-sizing-sm"
                        placeholder="использовать 70%"
                        disabled={!useBootstraping}
                        onChange={(e) => setBootstrapCoef(checkNumericInput(e, bootstrapCoef, true))}
                        value={bootstrapCoef}
                    />
                </InputGroup>
                <Form.Check
                    style={{fontSize: '0.8em'}}
                    type={'checkbox'}
                    label={`Не использовать bootstraping`}
                    id={`disabled-default-bootstrap`}
                    checked={!useBootstraping}
                    onChange={() => setUseBootstraping(!useBootstraping)}
                />

                <InputGroup size="sm" className="mb-3">
                    <InputGroup.Text id="inputGroup-sizing-sm">Random state</InputGroup.Text>
                    <Form.Control
                        aria-label="Small"
                        aria-describedby="inputGroup-sizing-sm"
                        placeholder='Полностью случайная модель'
                        onChange={(e) => setRandomState(checkNumericInput(e, randomState))}
                        value={randomState}
                    />
                </InputGroup>

                {
                    type == 'grad-boosting' && 
                    <InputGroup size="sm" className="mb-3">
                        <InputGroup.Text id="inputGroup-sizing-sm">learning rate</InputGroup.Text>
                        <Form.Control
                            aria-label="Small"
                            aria-describedby="inputGroup-sizing-sm"
                            placeholder='Задайте коэффициент'
                            onChange={(e) => setLearningRate(checkNumericInput(e, learningRate, true))}
                            value={learningRate}
                        />
                    </InputGroup>
                }
            </>
        )
    }

    return (
        <div style={{marginTop: '3%', display: 'flex', flexDirection: 'column',
                     border: '1px solid rgb(0,0,0, 0.1)', borderRadius: '5px', padding: '5px'}}>
            <span style={{fontSize: '1.2em', marginBottom: '10px'}}><bold>Гиперпараметры модели:</bold></span>
            <_Settings type={modelType}/>

        </div>
    )

}

export default ModelParams