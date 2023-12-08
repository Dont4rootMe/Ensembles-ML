import axios from "axios";

export class BadResponse {
    _validateResponse() {
        if (this.status == 403) {
            call_get(`${process.env.REACT_APP_BACKEND ?? 'http://localhost:8000'}/users/pullRole`).then((response) => {
                if (response instanceof Error) { console.log(response); return; }
                localStorage.setItem('role', response.data)
            })
        }
    }

    constructor(response) {
        this.status = response.response.status
        this.detail = response.response.data.detail
        this.message = response.message
    }
}

// Все сам писал

/**
 * Asynchronously makes a POST request to the specified path using axios HTTP client.
 * @param {string} path - The URL path of the endpoint to make the POST request to.
 * @param {object} params - parameters to pass
 * @returns {Promise} - Returns Response object if there was no any exceptions on the back side 
 *                      that resolves to the response data from the server, and return BadResponse object otherwise.
 */
export async function call_get(path, params = undefined) {
    try {
        return new Response(await axios.get(path, {
            params: params
        }))
    } catch (err) {
        console.log(err)
        return new BadResponse(err)
    }
}
/**
 * Asynchronously makes a POST request to the specified path using axios HTTP client.
 * @param {string} path - The URL path of the endpoint to make the POST request to.
 * @param {Object} body - The data to include in the request body.
 * @param {object} params - parameters to pass
 * @returns {Promise} -  Returns Response object if there was no any exceptions on the back side 
 *                       that resolves to the response data from the server, and return BadResponse object otherwise.
 */
export async function call_post(path, body, params={}) {
    try {
        return new Response(await axios.post(path, body, {
            params: params
        }))
    } catch (err) {
        console.log(err)
        return new BadResponse(err)
    }
}
