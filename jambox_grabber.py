import os
import threading

import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as chrome_options


class Jambox:

    sources = ["TVP1", "POLSAT", "TVN"]

    frames = {}

    def __init__(self, scaling_factor=1):

        self.scaling_factor = scaling_factor
        self.browser = self.setup_browser()
        self.jambox_login()

    def jambox_login(self):

        self.browser.get("https://go.jambox.pl/telewizja/ogladaj")
        self.browser.implicitly_wait(5)
        input(f"Log in and run stream for: {self.sources[0]}")

        for name in self.sources[1:]:
            self.browser.execute_script(
                'window.open("https://go.jambox.pl/telewizja/ogladaj", "_blank");'
            )
            self.browser.implicitly_wait(5)
            self.browser.switch_to_window(self.browser.window_handles[-1])
            input(f"Log in and run stream for: {name}")

        self.update_frames()
        print("Frames:", [frame.shape for frame in self.frames.values()])

    def setup_browser(self):

        # webdriver setup
        options = chrome_options()
        options.headless = False
        chromedriver = os.path.abspath("chromedriver")

        return webdriver.Chrome(executable_path=chromedriver, options=options)

    def close_browsers(self):
        for name in self.sources:
            self.browser.switch_to_window(self.browser.window_handles[0])
            print(f"Closing: {name}")
            self.browser.close()

    def update_frames(self):

        for window_id, name in enumerate(self.sources):

            self.browser.switch_to_window(self.browser.window_handles[window_id])
            self.get_frame(name)

    def get_frame(self, name):

        frame = self.browser.find_element_by_css_selector(
            "body > ng-include:nth-child(3) > div > div > div.video__right-click-overlay.ng-scope"
        ).screenshot_as_png

        if frame is not None:
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

            if self.scaling_factor > 1:

                height, width, _ = frame.shape

                new_height = height / self.scaling_factor
                new_width = width / self.scaling_factor

                frame = cv2.resize(frame, (int(new_width), int(new_height)))

            self.frames[name] = frame

    def is_ready(self):

        return len(self.frames) == len(self.sources)

