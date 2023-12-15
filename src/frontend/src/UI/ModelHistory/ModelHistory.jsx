import React, { useState } from 'react';
import { Card, Button, Badge, CloseButton, InputGroup, Form } from 'react-bootstrap';
import SmartPlot from '../SmartPlot/SmartPlot';
import { addWarning, addDanger, addSuccess } from '../../ToastFactory';
import { call_post, call_get, BadResponse } from '../../CALLBACKS';

const ModelHistory = ({ plate, deleteHistory }) => {

    const [showPrediction, setShowPrediction] = useState(false)
    const [predictionSet, setPredictionSet] = useState(null)

    const getFloatPrecision = (float, precision = 4) => {
        return (float.toFixed(precision))
    }

    const predictModel = async () => {
        if (!predictionSet) {
            addWarning('Предсказание данных', 'Укажите данные')
            return;
        }
        const formData = new FormData();
        formData.append("predict", predictionSet);
        const reply = await call_post(`http://localhost:8000/predict-model/${plate.key}`, formData)
        if (reply instanceof BadResponse) {
            addDanger('Что-то пошло не так', `Detail: ${reply.detail}`)
        } else {
            addSuccess('Предсказание модели', 'Файл успешно скачивается')

            var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reply.data));
            var dlAnchorElem = document.getElementById('downloadAnchorElem');
            dlAnchorElem.setAttribute("href", dataStr);
            dlAnchorElem.setAttribute("download", `predictions_model{plate.key}.json`);
            dlAnchorElem.click();
        }
    }

    const downloadModel = async () => {
        const reply = await call_get(`http://localhost:8000/download-model/${plate.key}`)
        if (reply instanceof BadResponse) {
            addDanger('Что-то пошло не так', `Detail: ${reply.detail}`)
        } else {
            addSuccess('Скачивание модели', 'Файл модели успешно скачивается')

            var blob = new Blob([reply.data], { type: '' });
            var link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            var fileName = `model_${plate.key}.db`;
            link.download = fileName;
            link.click();
        }
    }

    return (
        <Card key={plate.key} style={{ marginBottom: '5px' }}>
            <Card.Header as="h5" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex' }}>
                    <Badge bg="primary">{`model №${plate.key + 1}`}</Badge>
                    <span style={{ marginLeft: '5px' }}>{plate.history.model}</span>
                    {plate.history.dataset === 'dataset' ? <small style={{ marginLeft: '0.8em' }}>
                        {`target: ${plate.history.target}`}</small> : ''}
                </div>
                <CloseButton onClick={() => deleteHistory(plate.key)} />
            </Card.Header>

            {plate.history.trace && <SmartPlot data={plate.history.history} />}

            <Card.Body>
                <Card.Title style={{ marginBottom: '0px' }}>Параметры модели</Card.Title>
                <div style={{
                    width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr',
                    gridTemplateRows: '1fr 1fr 1fr' + plate.history.model !== 'Random forest' ? ' 1fr' : ''
                }}>
                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {plate.history.config.estimators && (`Число деревьев: ${plate.history.config.estimators}`)}
                    </Card.Text>
                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Глубина деревьев: ${plate.history.config.depth ? plate.history.config.depth : 'не ограничена'}`)}
                    </Card.Text>
                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {plate.history.config.fetSubsample && (`Число признаков: ${plate.history.config.fetSubsample}`)}
                    </Card.Text>

                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Стратегия ветвления: ${plate.history.config.useRandomSplit ? 'random' : 'best'}`)}
                    </Card.Text>
                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Random state: ${plate.history.config.randomState ? plate.history.config.randomState : 'случайная модель'}`)}
                    </Card.Text>
                    <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Bootstrap: ${plate.history.config.bootstrapCoef ? plate.history.config.bootstrapCoef : 'нет'}`)}
                    </Card.Text>

                    {plate.history.model !== 'Random forest' &&
                        <Card.Text style={{ marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                            {plate.history.config.learningRate && (`Learning rate: ${plate.history.config.learningRate}`)}
                        </Card.Text>}
                </div>

                <Card.Title style={{ marginBottom: '0px' }} >Конечные статистики</Card.Title>
                <Card.Text style={{ marginBottom: '0px', marginLeft: '1em' }}>
                    {plate.history.rmse && (`RMSE: ${getFloatPrecision(plate.history.rmse)}`)}
                </Card.Text>
                <Card.Text style={{ marginBottom: '0px', marginLeft: '1em' }}>
                    {plate.history.mae && (`MAE: ${getFloatPrecision(plate.history.mae)}`)}
                </Card.Text>
                <Card.Text style={{ marginBottom: '0px', marginLeft: '1em' }}>
                    {plate.history.mape && (`MAPE: ${getFloatPrecision(plate.history.mape)}`)}
                </Card.Text>
                <Card.Text style={{ marginBottom: '0px', marginLeft: '1em' }}>
                    {plate.history.r2 && (`R^2: ${getFloatPrecision(plate.history.r2)}`)}
                </Card.Text>
                <Card.Text style={{ marginBottom: '0px', marginLeft: '1em' }}>
                    {plate.history.r2 && (`Время обучения: ${getFloatPrecision(plate.history.time)} c.`)}
                </Card.Text>
            </Card.Body>
            {
                plate.history.dataset === 'dataset' &&
                <Card.Footer style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Button variant="primary" style={{ marginRight: '1em' }}
                        onClick={() => downloadModel()}
                    >Скачать модель</Button>
                    {
                        !showPrediction ? <Button variant="secondary" onClick={() => setShowPrediction(true)}>Предсказать</Button> :
                            <InputGroup style={{ width: '60%' }}>
                                <Form.Control
                                    type="file"
                                    placeholder="Файл для предсказания"
                                    onChange={(e) => setPredictionSet(e.target.files[0])}
                                />
                                <Button variant="outline-secondary"
                                    onClick={() => predictModel()}
                                >
                                    Начать
                                </Button>
                                <Button variant="outline-secondary" id="cancel-btn"
                                    onClick={() => { setShowPrediction(false); setPredictionSet(null) }}>
                                    Выйти
                                </Button>
                            </InputGroup>

                    }
                </Card.Footer>
            }
        </Card>
    )
}

export default ModelHistory