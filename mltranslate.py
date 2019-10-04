from selenium import webdriver
from selenium.webdriver.chrome.options import Options



class Translator():
    def __init__(self):
        firefox_options = Options()
        firefox_options.headless = True

        self.__driver = webdriver.Chrome(
            options=firefox_options,
            executable_path='/usr/lib/chromium-browser/chromedriver'
        )

    def translate(self, data):
        self.__driver.get("http://translate.google.com/#view=home&op=translate&sl=auto&tl=en&text=" + data)
        elem = self.__driver.find_elements_by_css_selector('.tlid-copy-target > .tlid-translation > span')

        if len(elem) == 0:
            return None

        return elem[0].text

    def __del__(self):
        self.__driver.close()



obj = Translator()

obj.translate("kaise ho")

