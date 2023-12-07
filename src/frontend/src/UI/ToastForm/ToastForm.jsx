import { useState } from 'react'
import { Toast } from 'react-bootstrap'

const ToastForm = ({ id, variant, head, body }) => {
    const [show, setShow] = useState(true)

    return (
        <Toast data-id={id} bg={variant} show={show} key={id} onClose={() => { setShow(false) }} delay={5000} autohide>
            <Toast.Header closeButton={true} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong>{head}</strong>
            </Toast.Header>
            <Toast.Body>
                <small>{body}</small>
            </Toast.Body>
        </Toast>
    )
}

export default ToastForm