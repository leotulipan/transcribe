import http.server
import socketserver
import os
import signal
import sys

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
    print("\nPress CTRL+C to stop the server")
    print("=" * 50)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped by user. Goodbye!")