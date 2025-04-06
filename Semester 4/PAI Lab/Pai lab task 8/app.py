import json
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

# Function to get models by make
def get_models_by_make(make):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/{make}?format=json"
    response = requests.get(url)
    data = response.json()
    results = data.get('Results', [])
    
    models = []
    if results:
        for item in results:
            models.append(item['Model_Name'])
    return models

# Function to get vehicle information by VIN
def get_vehicle_by_vin(vin):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
    response = requests.get(url)
    data = response.json()
    results = data.get('Results', [])
    
    vehicle_info = []
    for item in results:
        if item['Value']:
            vehicle_info.append({
                'variable': item['Variable'],
                'value': item['Value']
            })
    return vehicle_info

# HTTP request handler
class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        if self.path == '/get-models':
            self.handle_get_models()
        elif self.path == '/get-vehicle-info':
            self.handle_get_vehicle_info()

    def serve_html(self):
        try:
            with open("vehicle_info.html", "r") as file:
             html = file.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes(html, 'utf-8'))
        except IOError:
            self.send_error(404, "File Not Found")




    def handle_get_models(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        make = data.get('make')
        models = get_models_by_make(make)

        # Send the result as HTML response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response_html = f"<h2>Models for Make '{make}':</h2><ul>"
        if models:
            for model in models:
                response_html += f"<li>{model}</li>"
        else:
            response_html += "<li>No models found for this make.</li>"
        response_html += "</ul>"
        self.wfile.write(bytes(response_html, 'utf-8'))

    def handle_get_vehicle_info(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        vin = data.get('vin')
        vehicle_info = get_vehicle_by_vin(vin)

        # Send the result as HTML response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response_html = f"<h2>Vehicle Information for VIN: {vin}</h2><ul>"
        if vehicle_info:
            for info in vehicle_info:
                response_html += f"<li>{info['variable']}: {info['value']}</li>"
        else:
            response_html += "<li>No information found for this VIN.</li>"
        response_html += "</ul>"
        self.wfile.write(bytes(response_html, 'utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
