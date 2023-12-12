import axios from "axios";

export class Response {
    constructor(response) {
        this.data = response.data
    }
}

export class BadResponse {
    constructor(response) {
        this.status = response.response.status
        this.detail = response.response.data && response.response.data.detail
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
export async function call_post(path, body, params = {}) {
    try {
        return new Response(await axios.post(path, body, {
            params: params
        }))
    } catch (err) {
        console.log(err)
        return new BadResponse(err)
    }
}
