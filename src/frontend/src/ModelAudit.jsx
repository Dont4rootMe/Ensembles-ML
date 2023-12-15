import React, { useState, useEffect } from "react";
import ModelHistory from "./UI/ModelHistory/ModelHistory";



const ModelAudit = ({ modelHistoryLine, deleteHistory }) => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', paddingLeft: '20px' }}>
            <h1 style={{ fontSize: '2.3em', alignSelf: 'self-end', paddingRight: '3em' }}>История моделей</h1>
            <div style={{
                display: 'flex', flexDirection: 'column', width: '60%',
                position: 'absolute', overflowY: 'auto', height: '78%', top: '11%'
            }}>
                {
                    modelHistoryLine && modelHistoryLine.slice(0).reverse().map(plate => !plate.dontShow && <ModelHistory key={plate.key} plate={plate} deleteHistory={deleteHistory} />)
                }
            </div>
        </div>
    )
}

export default ModelAudit