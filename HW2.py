import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://bana290-assignment2.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

print(soup.prettify()[:3000])

