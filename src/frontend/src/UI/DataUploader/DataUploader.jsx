import React, { useState, useEffect } from "react";
import { Form, Button, InputGroup, Dropdown, DropdownButton } from "react-bootstrap";
import RangeSlider from "react-bootstrap-range-slider";
import { call_post, BadResponse } from "../../CALLBACKS";
import { addMessage, addSuccess, addWarning } from "../../ToastFactory";
import { addDanger } from "../../ToastFactory";


const DataUploader = ({ config, addHistory }) => {
    const [dataSet, setDataSet] = useState(undefined);
    const [validationSet, setValidationSet] = useState(null);
    const [targetIndex, setTargetIndex] = useState(undefined);


    const [targetList, setTargetList] = useState([])
    const [autoValid, setAutovalid] = useState(true)
    const [validSize, setValidSize] = useState(30)

    const _get_target_list = (delimiter) => {
        if (!dataSet) { return; }

        var reader = new FileReader();
        reader.onload = function () {
            const text = this.result;

            var line = text.split('\n')[0];
            setTargetList(line.split(delimiter));
        };
        reader.readAsText(dataSet);
    }

    useEffect(() => {
        setTargetIndex(targetList[0])
    }, [targetList])

    useEffect(() => {
        _get_target_list(',')
    }, [dataSet])

    const trainModel = async (trace) => {

        if (!dataSet) {
            addWarning('Загрузка данных', 'Вы не указали данные для обучения. Вам доступна опция синтетического датасета.')
            return;
        }
        if (!autoValid && !validationSet) {
            addWarning('Загрузка теста', 'Вам требуется указать данные, такого же образца, что и train, для теста модели')
            return;
        }
        if (!targetIndex) {
            addWarning('Выбор target переменной', 'Вам требуется указать переменную, считаемую таргетом')
            return
        }

        if (config.model === 'grad-boosting' && config.learningRate.length <= 0) {
            addDanger('Learning rate', 'Задайте явным образом learning rate')
            return
        }
        if (config.estimators.length <= 0) {
            addDanger('Число деревьев', 'Задайте явным образом параметр модели')
            return
        }

        addMessage('Обучение модели', 'Обучение модели уcпешно началось')

        let history = {
            dataset: 'dataset',
            trace: trace,
            target: targetIndex,
            test_size: autoValid ? validSize : null,
            config: { ...config }
        }

        const formData = new FormData();
        formData.append("train", dataSet);
        if (validSize) {
            formData.append('test', validationSet)
        }

        const _config = {
            ...config,
            test_size: autoValid ? validSize : null,
            target: targetIndex,
            synt_prefs: undefined
        }

        const reply = await call_post('http://localhost:8000/dataset-train', formData, { ..._config, trace: trace })
        if (reply instanceof BadResponse) {
            addDanger('Что-то пошло не так', `Detail: ${reply.detail}`)
        } else {
            addSuccess('Обучение модели', 'Модель обучилась успешно')
            addHistory(reply.data.number, { ...history, ...reply.data })
        }
    }


    return (
        <>
            <Form.Group controlId="formFile" size="sm" style={{ marginBottom: '5px' }}>
                <Form.Label style={{ marginBottom: '0px' }}>Выберете датасет для обучения</Form.Label>
                <Form.Control type="file" onChange={(e) => { _get_target_list(','); setDataSet(e.target.files[0]) }} />
            </Form.Group>

            <>
                <Form.Check
                    style={{ fontSize: '0.8em' }}
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
                    <Form.Group controlId="formFile" size="sm" style={{ marginBottom: '5px' }}>
                        <Form.Label style={{ marginBottom: '0px' }}>Выберете датасет для теста</Form.Label>
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
                        defaultValue={'sep-,'}
                    >
                        <Dropdown.Item value={'sep-,'} onClick={() => _get_target_list(',')}>delimiter:   ","</Dropdown.Item>
                        <Dropdown.Item value={'sep-;'} onClick={() => _get_target_list(';')}>delimiter:   ";"</Dropdown.Item>
                        <Dropdown.Item value={'sep-t'} onClick={() => _get_target_list('\t')}>delimiter:   "\t"</Dropdown.Item>

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

            <Dropdown >
                <Dropdown.Toggle variant="success" style={{ marginTop: '10px' }}>
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

export default DataUploader