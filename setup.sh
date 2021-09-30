mkdir -p ~/.streamlit

echo "[server]
headless = true
port = 8502
enableCORS = false
enableWebsocketCompression = false
" > ~/.streamlit/config.toml
