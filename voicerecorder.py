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

    def guess_type(self, path):
        """Override to set correct MIME type for HTML files"""
        if path.endswith('.html'):
            return 'text/html', None
        return super().guess_type(path)

def signal_handler(signum, frame):
    print("\nShutting down server...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for CTRL-C
    signal.signal(signal.SIGINT, signal_handler)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()
            httpd.server_close()