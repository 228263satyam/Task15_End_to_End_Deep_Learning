import os, unittest
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

class Task15APITests(unittest.TestCase):
    def test_home(self):
        r = requests.get(BASE_URL + "/", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertIn("service", r.json())

    def test_health(self):
        r = requests.get(BASE_URL + "/health", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["model_loaded"])

    def test_missing_file_is_rejected(self):
        r = requests.post(BASE_URL + "/predict", data={}, timeout=10)
        self.assertEqual(r.status_code, 400)

    def test_metrics(self):
        r = requests.get(BASE_URL + "/metrics", timeout=10)
        self.assertEqual(r.status_code, 200)
        self.assertIn("requests_total", r.json())

if __name__ == "__main__":
    unittest.main()
