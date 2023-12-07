import React, { useState, useEffect } from "react";
import { ToastContainer } from 'react-bootstrap';
import ToastForm from "./UI/ToastForm/ToastForm";

let id = 0

let setToasts;

let onClose;


export const addSuccess = (head, body) => {
    id++
    setToasts(
        <ToastForm id={id} key={id} variant="success" head={head} body={body} onClose={onClose} />
    )
}

export const addWarning = (head, body) => {
    id++
    setToasts(
        <ToastForm id={id} key={id} variant="warning" head={head} body={body} onClose={onClose} />
    )
}

export const addDanger = (head, body) => {
    id++
    setToasts(
        <ToastForm id={id} key={id} variant="danger" head={head} body={body} onClose={onClose} />
    )
}

export const addMessage = (head, body) => {
    id++
    setToasts(
        <ToastForm id={id} key={id} variant="light" head={head} body={body} onClose={onClose} />
    )
}


export const ToastList = ({ }) => {
    const [toastList, setToastList] = useState([])
    const [, forceUpdate] = useState()

    useEffect(
        () => {
            setToasts = (t) => { toastList.push(t); forceUpdate(t) }
        }, []
    )

    return (
        <ToastContainer style={{ position: "fixed", margin: "20px", right: "0px", bottom: '10px', margin: '5px' }}>
            {
                toastList.map(toast => toast)
            }
        </ToastContainer>
    )

}

export default ToastList