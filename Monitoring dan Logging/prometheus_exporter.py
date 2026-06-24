from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from http.server import HTTPServer, BaseHTTPRequestHandler
import psutil
import time
import random
import threading
import sys

CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage in percent')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Memory usage in percent')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage in percent')
NETWORK_SENT = Gauge('system_network_sent_bytes', 'Bytes sent over network')

REQUEST_COUNT = Counter('app_request_count_total', 'Total request count')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Number of active requests')
PREDICTION_PURCHASED = Counter('app_pred_purchased_total', 'Prediction result: Purchased')
PREDICTION_NOT_PURCHASED = Counter('app_pred_not_purchased_total', 'Prediction result: Not Purchased')
ERROR_COUNT = Counter('app_error_count_total', 'Total error count')

def collect_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
            DISK_USAGE.set(psutil.disk_usage('/').percent)
            NETWORK_SENT.set(psutil.net_io_counters().bytes_sent)
            time.sleep(5)
        except Exception as e:
            print(f"Error collecting system metrics: {e}")

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            start_time = time.time()
            ACTIVE_REQUESTS.inc()
            REQUEST_COUNT.inc()
            time.sleep(random.uniform(0.1, 0.5)) 

            if random.choice([True, False]):
                PREDICTION_PURCHASED.inc()
            else:
                PREDICTION_NOT_PURCHASED.inc()
            
            if random.random() < 0.1:
                ERROR_COUNT.inc()

            REQUEST_LATENCY.observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Data received successfully")
            
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    print("Memulai System Metrics Thread...")
    t = threading.Thread(target=collect_system_metrics)
    t.daemon = True
    t.start()

    print("Server Prometheus Exporter berjalan di port 8000...")
    print("Siap menerima POST di /predict dan GET di /metrics")
    
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleHandler)
    httpd.serve_forever()