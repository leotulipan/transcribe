import http.server
import socketserver
import os
import signal
import sys
import json
import urllib.request
import urllib.error
import urllib.parse

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable SharedArrayBuffer with COOP and COEP
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        if self.path == '/':
            self.path = '/voicerecorder.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_POST(self):
        # Proxy route for OpenAI API
        if self.path.startswith('/api/openai/'):
            print(f"Proxying OpenAI request: {self.path}")
            
            # Extract the actual OpenAI API path
            openai_path = self.path.replace('/api/openai', '')
            openai_url = f'https://api.openai.com{openai_path}'
            
            print(f"Forwarding to: {openai_url}")
            
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read request body
            request_body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Create a proper Request object with all headers
            headers = {key: value for key, value in self.headers.items() 
                      if key.lower() not in ('host', 'content-length')}
            
            # Create proxy request
            req = urllib.request.Request(
                openai_url,
                data=request_body,
                headers=headers,
                method='POST'
            )
            
            try:
                # Forward request to OpenAI
                with urllib.request.urlopen(req) as response:
                    # Read the response data
                    response_body = response.read()
                    
                    # Set the response status code
                    self.send_response(response.status)
                    
                    # Set CORS headers
                    self.send_header('Access-Control-Allow-Origin', '*')
                    
                    # Forward response headers
                    for header, value in response.getheaders():
                        if header.lower() not in ('transfer-encoding',):
                            self.send_header(header, value)
                    
                    # End headers
                    self.end_headers()
                    
                    # Write response body
                    self.wfile.write(response_body)
                    
                    print(f"OpenAI API response status: {response.status}")
            
            except urllib.error.HTTPError as e:
                # Handle HTTP errors from OpenAI
                print(f"OpenAI API error: {e.code} - {e.reason}")
                
                self.send_response(e.code)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                error_data = e.read()
                self.wfile.write(error_data)
                
            except Exception as e:
                # Handle other errors
                print(f"Proxy error: {str(e)}")
                
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({'error': str(e)}).encode('utf-8')
                self.wfile.write(error_response)
            
            return
        
        # Handle other POST requests normally
        return http.server.SimpleHTTPRequestHandler.do_POST(self)

    def guess_type(self, path):
        """Override to set correct MIME type for HTML files"""
        if path.endswith('.html'):
            return 'text/html', None
        return super().guess_type(path)

def signal_handler(signum, frame):
    print("\nShutting down server...")
    sys.exit(0)

class StoppableHTTPServer(socketserver.TCPServer):
    """HTTP server that can be stopped cleanly with CTRL-C"""
    allow_reuse_address = True
    
    def run(self):
        try:
            self.serve_forever()
        except KeyboardInterrupt:
            self.server_close()
            print("\nServer stopped successfully.")

if __name__ == "__main__":
    # Register signal handler for CTRL-C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create server with address reuse
    server = StoppableHTTPServer(("", PORT), Handler)
    
    # Print server information
    print(f"Voice Recorder Server running at http://localhost:{PORT}")
    print(f"API proxy enabled at http://localhost:{PORT}/api/openai/")
    print("\nPress CTRL+C to stop the server")
    print("=" * 50)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user. Goodbye!")