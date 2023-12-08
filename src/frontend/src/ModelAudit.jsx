import React, {useState, useEffect} from "react";
import ModelHistory from "./UI/ModelHistory/ModelHistory";



const ModelAudit = ({history}) => {

    return (
        <div style={{display: 'flex', flexDirection: 'column'}}>
            <h1 style={{ fontSize: '2.3em', alignSelf: 'self-end', paddingRight: '3em'}}>История моделей</h1>
            <div style={{display: 'flex', flexDirection: 'column'}}>
                {
                    history.map(history => <ModelHistory history={history} />)
                }
            </div>
        </div>
    )
}

export default ModelAudit