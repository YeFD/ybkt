import requests
from bs4 import BeautifulSoup as bs

file=open("C:/Users/94999/Desktop/scrapy/JavaScript.txt","w",encoding='UTF-8')
for i in range(1,15000):
    txurl="https://www.imooc.com/course/coursescore/id/36?page="+str(i)+".html"
    response=bs(requests.get(txurl).text,'html.parser')
    for k in response.find_all('div',{'class':'evaluation evaluate'}):
        file.write(k.p.text+" "+k.span.string+"\n")
file.close()

