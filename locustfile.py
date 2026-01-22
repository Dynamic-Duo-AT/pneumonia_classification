from locust import HttpUser, between, task

class MyUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def get_root(self):
        self.client.get("/")

    @task(10)
    def post_prediction(self):
        with open("tests/dummy_data/raw/train/NORMAL/IM-0119-0001.jpeg", "rb") as img_file:
            files = {"data": ("IM-0119-0001.jpeg", img_file, "image/jpeg")}
            self.client.post("/pred/", files=files)