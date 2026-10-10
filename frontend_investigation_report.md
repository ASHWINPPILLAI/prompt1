# Investigation Report: Old Frontend vs New Frontend API Layer

Here is the exact behavior of the **old frontend** before it was replaced:

### a) Endpoint URL and HTTP Method
- **URL**: `/api/query_stream` (relative to the host).
- **Method**: `POST`

### b) Request Body Fields
The old frontend sent a JSON body containing three fields:
- `question` (string): The user's input.
- `prompt_id` (string): The selected version (e.g., "v0_baseline").
- `history` (array): The conversation history in the format `[{role: 'user', content: '...'}, ...]`.

*(Note: The new UI expects `{ question, version, params: {...} }`. The old API used `prompt_id` instead of `version` and passed `history`).*

### c) Response Format & Streaming
- **Format**: Server-Sent Events (SSE). It requested `text/event-stream`.
- **Streamed**: Yes. It read the stream line-by-line using a `TextDecoder` and `response.body.getReader()`.
- **Payload**: Each chunk was prefixed with `data: ` and was a JSON string like `{"content": "..."}` or `{"error": "..."}`. The stream ended when the server sent `data: [DONE]`.

### d) "Network Mode" (OpenRouter Fallback)
- **How it worked**: In the old frontend, the "Network Mode" selector was purely cosmetic! It was a `<select>` with a single hardcoded option (`<option>OpenRouter Fallback</option>`) and it didn't even have an `id`. 
- **What it sent**: It sent nothing. The `askQuestion()` JavaScript function never read its value or passed it to the backend.

### e) Backend Startup & Port
- **Startup**: The backend is started via `python src/policypilot/app/server.py [port]`.
- **Port**: It listens on port `8501` by default (unless specified otherwise).

### f) CORS Settings / Proxy
- **CORS**: There were no CORS headers (`Access-Control-Allow-Origin`) in the backend's Python HTTP server because the frontend (`index.html`) was served directly by the *same* Python backend at the root path `/`. Because it was same-origin, CORS was not needed.

---
**Status**: Awaiting your approval. Once approved, I will revert the backend code to its original streaming state (so it remains untouched as requested), and entirely rewrite the new frontend's `API.ask` logic to adapt to this old streaming endpoint!
