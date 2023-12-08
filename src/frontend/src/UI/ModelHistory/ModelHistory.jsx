import React, {useState} from 'react';
import { Card, Button, Badge, CloseButton } from 'react-bootstrap';
import SmartPlot from '../SmartPlot/SmartPlot';

const ModelHistory = ({plate, deleteHistory}) => {
    console.log(plate)
    const getFloatPrecision = (float, precision=4) => {
        return (float.toFixed(precision))
    }

    return (
        <Card key={plate.key}>
            <Card.Header as="h5" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                <div style={{display: 'flex'}}>
                    <Badge bg="primary">{`model №${plate.key + 1}`}</Badge>
                    <span style={{marginLeft: '5px'}}>{plate.history.model}</span>
                </div>
                <CloseButton onClick={() => deleteHistory(plate.key)}/>
            </Card.Header>

            {plate.history.trace && <SmartPlot data={plate.history.history}/>}

            <Card.Body>
                <Card.Title style={{marginBottom: '0px'}}>Данные модели</Card.Title>
                <div style={{width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr',
                             gridTemplateRows: '1fr 1fr 1fr' + plate.history.model !== 'Random forest' ? ' 1fr' : ''}}>
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {plate.history.config.estimators && (`Число деревьев: ${plate.history.config.estimators}`)}
                    </Card.Text>
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Глубина деревьев: ${plate.history.config.depth ? plate.history.config.depth : 'не ограничена'}`)}
                    </Card.Text>
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {plate.history.config.fetSubsample && (`Число признаков: ${plate.history.config.fetSubsample}`)}
                    </Card.Text>

                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Стратегия ветвления: ${plate.history.config.useRandomSplit ? 'random' : 'best'}`)}
                    </Card.Text>
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Random state: ${plate.history.config.randomState ? plate.history.config.randomState : 'случайная модель'}`)}
                    </Card.Text>
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {(`Bootstrap: ${plate.history.config.bootstrapCoef ? plate.history.config.bootstrapCoef : 'нет'}`)}
                    </Card.Text>

                    {plate.history.model !== 'Random forest' && 
                    <Card.Text style={{marginBottom: '0px', marginLeft: '1em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {plate.history.config.learningRate && (`Learning rate: ${plate.history.config.learningRate}`)}
                    </Card.Text>}
                </div>

                <Card.Title style={{marginBottom: '0px'}} >Конечные статистики</Card.Title>
                <Card.Text style={{marginBottom: '0px', marginLeft: '1em'}}>
                    {plate.history.mse && (`MSE: ${getFloatPrecision(plate.history.mse)}`)}
                </Card.Text>
                <Card.Text style={{marginBottom: '0px', marginLeft: '1em'}}>
                    {plate.history.mape && (`MAPE: ${getFloatPrecision(plate.history.mape)}`)}
                </Card.Text>
                <Card.Text style={{marginBottom: '0px', marginLeft: '1em'}}>
                    {plate.history.r2 && (`R^2: ${getFloatPrecision(plate.history.r2)}`)}
                </Card.Text>
            </Card.Body>
        </Card>
    )
}

export default ModelHistory